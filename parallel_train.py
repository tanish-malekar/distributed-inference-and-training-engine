"""Multi-process training driver; the parallel counterpart of train.run (same data, settings,
log format and checkpoint cadence). Launch with torchrun; see README.md."""

import os

import torch
import torch.distributed as dist
from transformers import AutoConfig

from data import fingerprint, prepare_data, training_batch
from distributed import init_parallel, rank_zero_first, run_or_exit, validate_layout
from model import load_tokenizer
from parallel_checkpoint import restore_training, save_checkpoint
from parallel_engine import ParallelEngine
from parallel_model import create_stage, resolve_source


def run_parallel(args):
    check_layout(args)
    ctx = init_parallel(args.dp, args.tp, args.pp, args.device)
    run_or_exit(lambda: train_parallel(args, ctx))


# Train inside an already-initialized process group (run_parallel and the smoke test share this).
def train_parallel(args, ctx):
    torch.manual_seed(args.seed)
    source = args.resume or args.model
    # Downloads happen once, on rank 0, before the other ranks read the cache.
    rank_zero_first(ctx, lambda: resolve_source(source))
    tokenizer = rank_zero_first(ctx, lambda: load_tokenizer(source))
    config = AutoConfig.from_pretrained(source)
    if args.sequence_length > config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the model's context window")
    train_data, validation_data = rank_zero_first(ctx, lambda: prepare_data(
        tokenizer, args.sequence_length, args.train_rows, args.validation_rows,
        args.train_file, args.validation_file))
    stage = create_stage(source, ctx, shard=args.dp_shard)
    # foreach=False: the multi-tensor path (CUDA's default) cannot mix the plain tensors (norm
    # weights) and DTensors (sharded weights) that tensor parallelism produces.
    optimizer = torch.optim.AdamW(stage.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=False)
    # The layout is deliberately NOT part of the settings: a run may resume under another one.
    settings = {key: getattr(args, key) for key in (
        "batch_size", "sequence_length", "learning_rate", "weight_decay", "max_grad_norm", "seed")}
    settings.update(train_data=fingerprint(train_data), validation_data=fingerprint(validation_data))
    completed = restore_training(args.resume, stage, optimizer, settings, ctx) if args.resume else 0
    if args.steps <= completed:
        raise ValueError("--steps must exceed the checkpoint's completed step count")
    engine = ParallelEngine(stage, ctx, args.batch_size, args.microbatches, args.pp_schedule)
    # A tied matrix lives on two stages when pp > 1 but is one parameter of the model.
    duplicate = stage.lm_head.weight if (engine.tied and ctx.is_last_stage) else None
    count = torch.tensor(sum(p.numel() for p in stage.parameters() if p is not duplicate),
                         device=ctx.device)
    if ctx.pp > 1:
        dist.all_reduce(count, group=ctx.pp_group)
    parameters = count.item()
    log(ctx, f"Device: {ctx.device} | layout: dp={ctx.dp} tp={ctx.tp} pp={ctx.pp} "
             f"| trainable parameters: {parameters:,} / {parameters:,}")
    log(ctx, f"Packed sequences: train={len(train_data['input_ids'])}, "
             f"validation={len(validation_data['input_ids'])}")
    log(ctx, f"Initial validation loss: {engine.evaluate(validation_data, args.batch_size):.4f}")
    for step in range(completed, args.steps):
        batch = training_batch(train_data, step, args.batch_size, args.seed, torch.device("cpu"))
        loss, tokens, grad_norm = engine.train_step(batch, optimizer, args.max_grad_norm)
        log(ctx, f"Step {step + 1}: train_loss={loss:.4f}, target_tokens={tokens}, grad_norm={grad_norm:.4f}")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            validation_loss = engine.evaluate(validation_data, args.batch_size)
            checkpoint = save_checkpoint(stage, tokenizer, optimizer, step + 1, settings, args.output, ctx)
            log(ctx, f"Validation loss={validation_loss:.4f} | saved {checkpoint}")


# Fail on a layout the model cannot support before any process allocates memory.
def check_layout(args):
    world = int(os.environ.get("WORLD_SIZE", 1))
    source = args.resume or args.model
    validate_layout(AutoConfig.from_pretrained(source), args.dp, args.tp, args.pp, world)


def log(ctx, message):
    if ctx.is_main:
        print(message, flush=True)
