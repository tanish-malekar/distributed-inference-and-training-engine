"""One pipeline stage of the Hugging Face Llama model, with tensor and data parallelism.

The stage keeps the ORIGINAL Hugging Face submodules and parameter names
(model.embed_tokens, model.layers.N..., model.norm, lm_head), so checkpoints map 1:1
between any parallel layout and the single-device model. Parallelism comes from PyTorch:
DTensor tensor parallelism, FSDP2 replicate/fully_shard, and torch.distributed.checkpoint.
"""

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


# Whole pipeline: meta build -> tensor parallel -> data parallel -> allocate -> load.
# training=False skips the data-parallel wrapper (inference replicas are independent) and
# freezes the weights. whole=True builds the complete HF model (requires pp == 1).
def create_stage(source, ctx, shard=False, gather_logits=False, load=True, training=True, whole=False):
    if whole and ctx.pp != 1:
        raise ValueError("The complete model can only be built with --pp 1")
    stage = build_stage(source, ctx, whole)
    apply_tensor_parallel(stage, ctx, gather_logits)
    retie_embeddings(stage)
    if training:
        apply_data_parallel(stage, ctx, shard)
    materialize(stage, ctx)
    if ctx.dp == 1 or not training:
        retie_embeddings(stage)
    if _first(stage) and _last(stage) and stage.config.tie_word_embeddings:
        if stage.lm_head.weight is not stage.model.embed_tokens.weight:
            raise RuntimeError("Tied embedding and output weights are no longer one parameter")
    if load:
        load_stage_weights(stage, source)
    stage.requires_grad_(training)
    return stage


# Build this rank's stage on the meta device: shapes and names only, no memory.
# whole=True returns the complete Hugging Face model instead (valid only when pp == 1).
def build_stage(source, ctx, whole=False):
    config = AutoConfig.from_pretrained(source)
    config._attn_implementation = "eager"
    config.use_cache = False
    if ctx.tp > 1 and config.pad_token_id is not None:
        # DTensor embeddings do not implement padding_idx's zero-gradient row.
        raise ValueError("Tensor parallelism requires a model config without pad_token_id")
    with torch.device("meta"):
        full = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    if whole:
        return full
    layers = split_layers(config.num_hidden_layers, ctx.pp)[ctx.pp_rank]
    return LlamaStage(full, layers, ctx.is_first_stage, ctx.is_last_stage)


class LlamaStage(nn.Module):
    def __init__(self, full, layers, first, last):
        super().__init__()
        self.config, self.first, self.last = full.config, first, last
        self.first_layer = layers[0]  # cache position and mask are sized against this layer
        self.model = nn.Module()
        if first:
            self.model.embed_tokens = full.model.embed_tokens
        # A ModuleDict keyed by the original index keeps names such as model.layers.17.mlp...
        self.model.layers = nn.ModuleDict({str(i): full.model.layers[i] for i in layers})
        self.model.rotary_emb = full.model.rotary_emb
        if last:
            self.model.norm = full.model.norm
            self.lm_head = full.lm_head

    # Token IDs (first stage) or hidden states in, hidden states or logits (last stage) out.
    # Pipeline training passes only the tensor; inference may add a mask, positions and a cache.
    def forward(self, x, attention_mask=None, position_ids=None, past_key_values=None):
        hidden = self.model.embed_tokens(x) if self.first else x
        if position_ids is None:
            past = past_key_values.get_seq_length(self.first_layer) if past_key_values is not None else 0
            position_ids = (torch.arange(hidden.shape[1], device=hidden.device) + past).unsqueeze(0)
        mask = create_causal_mask(
            config=self.config, inputs_embeds=hidden, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids, layer_idx=self.first_layer)
        embeddings = self.model.rotary_emb(hidden, position_ids=position_ids)
        for layer in self.model.layers.values():
            hidden = layer(hidden, attention_mask=mask, position_embeddings=embeddings,
                           position_ids=position_ids, past_key_values=past_key_values)
        if self.last:
            return self.lm_head(self.model.norm(hidden))
        return hidden


# Contiguous, balanced layer ranges; earlier stages take the remainder.
def split_layers(num_layers, pp):
    base, extra = divmod(num_layers, pp)
    ranges, start = [], 0
    for stage in range(pp):
        size = base + (1 if stage < extra else 0)
        ranges.append(range(start, start + size))
        start += size
    return ranges


# Column-parallel then row-parallel pairs need one all-reduce per block; vocabulary is split too.
def apply_tensor_parallel(stage, ctx, gather_logits=False):
    if ctx.tp == 1:
        return
    mesh = ctx.mesh["tp"]
    layer_plan = {
        "self_attn.q_proj": ColwiseParallel(), "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(), "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(), "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
    for layer in _layers(stage):
        parallelize_module(layer, mesh, layer_plan)
    if _first(stage):
        parallelize_module(stage.model, mesh, {
            "embed_tokens": RowwiseParallel(input_layouts=Replicate())})
    if _last(stage):
        # Training keeps logits split over the vocabulary for the vocabulary-parallel loss;
        # inference gathers them so every rank can pick the next token.
        parallelize_module(stage, mesh, {"lm_head": ColwiseParallel(
            output_layouts=Replicate() if gather_logits else Shard(-1), use_local_output=True)})


# Module-by-module sharding and meta-to-device allocation both split a tied parameter
# into independent copies; make them one parameter again. One stage holds both ends only
# when pp == 1.
def retie_embeddings(stage):
    if _first(stage) and _last(stage) and stage.config.tie_word_embeddings:
        stage.lm_head.weight = stage.model.embed_tokens.weight


# Default is replicated data parallel (gradient all-reduce); --dp-shard shards state (FSDP).
def apply_data_parallel(stage, ctx, shard):
    if ctx.dp == 1:
        return
    wrap, mesh = (fully_shard if shard else replicate), ctx.mesh["dp"]
    options = {"reshard_after_forward": ctx.pp == 1} if shard else {}
    for layer in _layers(stage):
        wrap(layer, mesh=mesh, **options)
    wrap(stage, mesh=mesh, **options)


# Allocate real storage on the device and rebuild buffers that are not in state_dicts.
def materialize(stage, ctx):
    stage.to_empty(device=ctx.device)
    with torch.device(ctx.device):
        stage.model.rotary_emb = LlamaRotaryEmbedding(config=stage.config)


# Every rank reads only the slices of the safetensors checkpoint that it owns.
def load_stage_weights(stage, source):
    directory = resolve_source(source)
    state = dict(stage.state_dict())
    if stage.config.tie_word_embeddings and _last(stage) and not _first(stage):
        # The file stores the tied matrix once, under the embedding name.
        state["model.embed_tokens.weight"] = state.pop("lm_head.weight")
    elif stage.config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    dcp.load(state, storage_reader=HuggingFaceStorageReader(str(directory)))


def resolve_source(source):
    if Path(source).is_dir():
        return Path(source)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(source, allow_patterns=["*.safetensors", "*.json"]))


# The helpers below also accept a plain Hugging Face LlamaForCausalLM (used for single-stage
# inference so HF generate and its KV cache can run unchanged).
def _layers(model):
    layers = model.model.layers
    return list(layers.values()) if isinstance(layers, nn.ModuleDict) else list(layers)


def _first(model):
    return getattr(model, "first", True)


def _last(model):
    return getattr(model, "last", True)
