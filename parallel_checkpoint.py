"""Save and resume a (pp, dp, tp) training run with torch.distributed.checkpoint (DCP).

step-NNNNNN/ holds
  model.safetensors (+ index), config.json, tokenizer   Hugging Face format, consolidated from
                                                        every rank's shards; loads with
                                                        from_pretrained / load_model / inference.py
  optimizer/                                            DCP AdamW state; DCP reshards on load, so a
                                                        run may resume under a different dp/tp/pp
  training_state.pt                                     step, settings, data fingerprints, layout
  rng/rank-NNNNN.pt                                     per-rank RNG, restored when the layout is unchanged
"""

import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict)

EMBEDDING, OUTPUT = "model.embed_tokens.weight", "lm_head.weight"
# Flattened keys ("state.<param>.exp_avg", "param_groups.<param>.lr") are unique per parameter.
# The default nested form keys parameter groups by index, so different pipeline stages would
# overwrite each other's group 0 (and its parameter list) in the shared checkpoint.
FLAT = StateDictOptions(flatten_optimizer_state_dict=True)


def save_checkpoint(stage, tokenizer, optimizer, step, settings, output, ctx):
    root = Path(output)
    target = root / f"step-{step:06d}"
    plan = None
    if ctx.is_main:
        if target.exists():
            plan = ("exists", str(target))
        else:
            root.mkdir(parents=True, exist_ok=True)
            plan = ("ok", tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    kind, path = _agree(plan, ctx)
    if kind == "exists":
        raise FileExistsError(f"Checkpoint already exists: {path}; choose another --output")
    temporary = Path(path)
    try:
        dcp.save(hf_state(stage), storage_writer=HuggingFaceStorageWriter(
            path=str(temporary), save_distributed=True, enable_consolidation=True))
        dcp.save({"optim": _canonical_optimizer_state(stage, optimizer)},
                 checkpoint_id=str(temporary / "optimizer"))
        (temporary / "rng").mkdir(exist_ok=True)
        rng = {"cpu": torch.get_rng_state()}
        if ctx.device.type == "cuda":
            rng["cuda"] = torch.cuda.get_rng_state(ctx.device)
        torch.save(rng, temporary / "rng" / f"rank-{ctx.rank:05d}.pt")
        dist.barrier()
        if ctx.is_main:
            stage.config.save_pretrained(temporary)
            tokenizer.save_pretrained(temporary)
            shutil.rmtree(temporary / "sharded", ignore_errors=True)  # per-rank pieces; consolidated already
            torch.save({"step": step, "settings": settings, "layout": ctx.layout,
                        "device_type": ctx.device.type}, temporary / "training_state.pt")
            os.rename(temporary, target)  # Publish only after every file is written.
        dist.barrier()
    except BaseException:
        if ctx.is_main:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


# Weights come from the checkpoint's Hugging Face files (create_stage(source=checkpoint)); this
# restores the optimizer history, RNG and step counter around them.
def restore_training(checkpoint, stage, optimizer, settings, ctx):
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    if state["settings"] != settings:
        raise ValueError("Resume settings/data changed; use the same data and training settings")
    if state["device_type"] != ctx.device.type:
        raise ValueError("Resume on the same device type; use --model CHECKPOINT for weights-only loading")
    target = _canonical_optimizer_state(stage, optimizer)  # also creates AdamW's state tensors
    dcp.load({"optim": target}, checkpoint_id=str(checkpoint / "optimizer"))
    if _holds_output_copy(stage):
        _rename_optimizer_key(target, EMBEDDING, OUTPUT)
    set_optimizer_state_dict(stage, optimizer, optim_state_dict=target, options=FLAT)
    if state["layout"] == ctx.layout:
        rng = torch.load(checkpoint / "rng" / f"rank-{ctx.rank:05d}.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        if ctx.device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], ctx.device)
    return state["step"]


# Weights in Hugging Face form: a tied output matrix is stored once, under the embedding name.
def hf_state(stage):
    state = dict(stage.state_dict())
    if stage.config.tie_word_embeddings:
        state.pop(OUTPUT, None)
    return state


def _canonical_optimizer_state(stage, optimizer):
    state = get_optimizer_state_dict(stage, optimizer, options=FLAT)
    if _holds_output_copy(stage):
        _rename_optimizer_key(state, OUTPUT, EMBEDDING)
    return state


def _holds_output_copy(stage):
    return stage.config.tie_word_embeddings and stage.last and not stage.first


# When pp > 1 the last stage's copy of the tied matrix is called lm_head.weight, while every
# other layout calls it model.embed_tokens.weight. Store one name so layouts are interchangeable.
def _rename_optimizer_key(flat, old, new):
    for key in list(flat):
        section, _, name = key.partition(".")  # "state" or "param_groups", then the parameter
        if name == old or name.startswith(old + "."):
            flat[f"{section}.{new}{name[len(old):]}"] = flat.pop(key)


def _agree(value, ctx):
    holder = [value]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]
