"""End-to-end check of one parallel layout: train, checkpoint, resume, then generate.

Launch under torchrun with dp*tp*pp processes (scripts/context3.sh smoke does this on
Context3 after the GPU check and your approval). To judge correctness, run train.py on
one device with the same settings and compare the printed losses:
  --steps 3 --batch-size 2*dp*microbatches --sequence-length 128 --train-rows 100
  --validation-rows 20 --eval-every 25
(microbatches defaults to --pp).
"""

import argparse
import faulthandler
import os
from pathlib import Path
import sys

# Allow launching from the project root as: python scripts/parallel_smoke_test.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig

from distributed import init_parallel, run_or_exit, validate_layout
from model import DEFAULT_MODEL, load_tokenizer
from parallel_generate import generate_distributed
from parallel_model import create_stage
from parallel_train import train_parallel


def main():
    # DITE_DUMP_STACKS=SECONDS prints every thread's stack once, to show where a hung rank waits.
    if int(os.environ.get("DITE_DUMP_STACKS", "0")) > 0:
        faulthandler.dump_traceback_later(int(os.environ["DITE_DUMP_STACKS"]), exit=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--microbatches", type=int)
    parser.add_argument("--pp-schedule", choices=["1f1b", "gpipe"], default="1f1b")
    parser.add_argument("--dp-shard", action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Choose a fresh output directory")
    args.microbatches = args.microbatches or args.pp
    settings = argparse.Namespace(
        model=DEFAULT_MODEL, device=args.device, steps=2, batch_size=2 * args.dp * args.microbatches,
        sequence_length=128, learning_rate=5e-5, weight_decay=0.01, max_grad_norm=1.0,
        eval_every=25, train_rows=100, validation_rows=20, train_file=None, validation_file=None,
        seed=42, output=args.output, resume=None, dp=args.dp, tp=args.tp, pp=args.pp,
        microbatches=args.microbatches, pp_schedule=args.pp_schedule, dp_shard=args.dp_shard,
    )
    validate_layout(AutoConfig.from_pretrained(DEFAULT_MODEL), args.dp, args.tp, args.pp,
                    int(os.environ.get("WORLD_SIZE", 1)))
    ctx = init_parallel(args.dp, args.tp, args.pp, args.device)

    def work():
        note = lambda text: print(text, flush=True) if ctx.is_main else None
        note("TEST 1: two full fine-tuning steps and checkpoint saving")
        train_parallel(settings, ctx)
        settings.resume = str(Path(args.output) / "step-000002")
        settings.steps = 3
        note("TEST 2: restore the optimizer and data cursor, continue to step three")
        train_parallel(settings, ctx)
        note("TEST 3: load the fine-tuned checkpoint and generate")
        checkpoint = str(Path(args.output) / "step-000003")
        model = create_stage(checkpoint, ctx, gather_logits=True, training=False, whole=args.pp == 1)
        texts = generate_distributed(
            model, load_tokenizer(checkpoint), ["The purpose of machine learning is"], ctx,
            max_new_tokens=24, microbatches=1)
        if not texts[0].strip():
            raise RuntimeError("Generation produced empty text")
        note(texts[0])
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
            note(f"Rank 0 peak GPU allocated memory: {torch.cuda.max_memory_allocated(ctx.device) / 2**30:.2f} GiB")
        note("PASS: parallel training, validation, checkpoint save/resume, and inference")

    run_or_exit(work)


if __name__ == "__main__":
    main()
