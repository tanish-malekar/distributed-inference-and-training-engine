"""Text generation under data, tensor and pipeline parallelism.

pp == 1: the complete Hugging Face model (tensor-parallel DTensor weights) runs through HF
         `generate` and its KV cache unchanged; data parallelism splits the prompts.
pp  > 1: HF `generate` cannot drive layers living in different processes, so each stage
         runs its layers with its own HF DynamicCache and hands hidden states to the next
         stage. The last stage chooses tokens and shares them with every stage.
"""

import torch
import torch.distributed as dist
from transformers import DynamicCache, TemperatureLogitsWarper, TopPLogitsWarper


# Every rank calls this with the same prompts; returns the completions in prompt order.
def generate_distributed(model, tokenizer, prompts, ctx, max_new_tokens=64, sample=False,
                         temperature=0.8, top_p=0.9, seed=42, microbatches=1):
    config = model.config
    longest = max(len(tokenizer(p)["input_ids"]) for p in prompts)
    if longest + max_new_tokens > config.max_position_embeddings:
        raise ValueError("Prompt plus generated tokens exceeds the model's context window")
    mine = list(range(ctx.dp_rank, len(prompts), ctx.dp))  # prompts are dealt out to dp replicas
    texts = []
    if mine:
        batch = [prompts[i] for i in mine]
        # Tensor-parallel peers must draw identical samples, so the seed depends only on dp_rank.
        torch.manual_seed(seed + ctx.dp_rank)
        if ctx.pp == 1:
            texts = _generate_with_hf(model, tokenizer, batch, ctx, max_new_tokens, sample, temperature, top_p)
        else:
            texts = _generate_pipelined(model, tokenizer, batch, ctx, max_new_tokens, sample,
                                        temperature, top_p, microbatches)
    pieces = [None] * ctx.dp
    if ctx.dp > 1:
        dist.all_gather_object(pieces, (mine, texts), group=ctx.dp_group)
    else:
        pieces = [(mine, texts)]
    ordered = [None] * len(prompts)
    for indices, chunk in pieces:
        for index, text in zip(indices, chunk):
            ordered[index] = text
    return ordered


@torch.inference_mode()
def _generate_with_hf(model, tokenizer, prompts, ctx, max_new_tokens, sample, temperature, top_p):
    model.eval()
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(ctx.device)
    options = {"temperature": temperature, "top_p": top_p} if sample else {}
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=sample, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, **options)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Split prompts into at most `count` non-empty groups, keeping order.
def _chunks(items, count):
    count = max(1, min(count, len(items)))
    size, extra = divmod(len(items), count)
    groups, start = [], 0
    for index in range(count):
        end = start + size + (1 if index < extra else 0)
        groups.append(items[start:end])
        start = end
    return groups


@torch.inference_mode()
def _generate_pipelined(stage, tokenizer, prompts, ctx, max_new_tokens, sample, temperature, top_p, microbatches):
    stage.eval()
    device, config = ctx.device, stage.config
    groups = _chunks(prompts, microbatches)
    encoded = [tokenizer(group, return_tensors="pt", padding=True, padding_side="left") for group in groups]
    ids = [e["input_ids"].to(device) for e in encoded]
    masks = [e["attention_mask"].to(device) for e in encoded]
    caches = [DynamicCache(config=config) for _ in groups]
    sizes = [len(group) for group in groups]
    finished = [torch.zeros(size, dtype=torch.bool, device=device) for size in sizes]
    generated = [[] for _ in groups]
    warpers = []
    if sample:
        warpers = [TemperatureLogitsWarper(temperature), TopPLogitsWarper(top_p)]
    last_tokens = [None] * len(groups)
    for step in range(max_new_tokens):
        sends, chosen = [], []
        for m in range(len(groups)):
            inputs = ids[m] if step == 0 else last_tokens[m]
            positions = (masks[m].cumsum(-1) - 1).clamp(min=0)[:, -inputs.shape[1]:]
            if ctx.is_first_stage:
                hidden = inputs
            else:
                hidden = torch.empty(inputs.shape[0], inputs.shape[1], config.hidden_size, device=device)
                dist.recv(hidden, group_src=ctx.pp_rank - 1, group=ctx.pp_group)
            output = stage(hidden, attention_mask=masks[m], position_ids=positions,
                           past_key_values=caches[m])
            if ctx.is_last_stage:
                logits = output[:, -1, :]
                for warper in warpers:
                    logits = warper(None, logits)
                chosen.append(torch.multinomial(logits.softmax(-1), 1).squeeze(-1) if sample
                              else logits.argmax(-1))
            else:
                # Sending is asynchronous so this stage can start the next microbatch.
                sends.append(dist.isend(output.contiguous(), group_dst=ctx.pp_rank + 1, group=ctx.pp_group))
        for send in sends:
            send.wait()
        # One collective per step: every stage learns the new tokens of every microbatch.
        tokens = (torch.cat(chosen) if ctx.is_last_stage
                  else torch.empty(sum(sizes), dtype=torch.long, device=device))
        dist.broadcast(tokens, group_src=ctx.pp - 1, group=ctx.pp_group)
        for m, piece in enumerate(tokens.split(sizes)):
            piece = torch.where(finished[m], torch.full_like(piece, tokenizer.pad_token_id), piece)
            finished[m] |= piece == tokenizer.eos_token_id
            generated[m].append(piece)
            last_tokens[m] = piece.unsqueeze(1)
            masks[m] = torch.cat([masks[m], torch.ones_like(masks[m][:, :1])], dim=1)
        if all(bool(f.all()) for f in finished):
            break
    texts = []
    for m in range(len(groups)):
        full = torch.cat([ids[m], torch.stack(generated[m], dim=1)], dim=1)
        texts.extend(tokenizer.batch_decode(full, skip_special_tokens=True))
    return texts
