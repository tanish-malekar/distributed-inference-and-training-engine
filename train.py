"""Full fine-tuning of a pretrained causal LM on one device. Start reading here."""

import argparse
import math

import torch

from data import fingerprint, prepare_data, training_batch
from engine import choose_device, evaluate, restore_training, save_checkpoint, train_step
from model import DEFAULT_MODEL, load_model


# Read the run settings; --steps is the TOTAL desired optimizer-step count.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--train-rows", type=int, default=1000)
    parser.add_argument("--validation-rows", type=int, default=200)
    parser.add_argument("--train-file", help="Optional UTF-8 text file, one document per line")
    parser.add_argument("--validation-file", help="Separate held-out text file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/smollm2")
    parser.add_argument("--resume", help="Checkpoint directory including training_state.pt")
    args = parser.parse_args()
    for name in ("steps", "batch_size", "eval_every", "train_rows", "validation_rows"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if not (all(math.isfinite(v) for v in (args.learning_rate, args.max_grad_norm, args.weight_decay))
            and args.learning_rate > 0 and args.max_grad_norm > 0 and args.weight_decay >= 0):
        parser.error("Learning rate/gradient limit must be positive; weight decay must be nonnegative")
    if bool(args.train_file) != bool(args.validation_file):
        parser.error("Provide both --train-file and --validation-file, or neither")
    run(args)


# Orchestrate data, local steps, evaluation, and checkpoints; no communication yet.
def run(args):
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    model, tokenizer = load_model(args.resume or args.model, device)
    if args.sequence_length > model.config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the model's context window")
    train_data, validation_data = prepare_data(
        tokenizer, args.sequence_length, args.train_rows, args.validation_rows,
        args.train_file, args.validation_file,
    )
    # All parameters go into AdamW: this is full fine-tuning, not adapter training.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # A matching fingerprint prevents silently resuming against different tokens.
    settings = {key: getattr(args, key) for key in (
        "batch_size", "sequence_length", "learning_rate", "weight_decay", "max_grad_norm", "seed")}
    settings.update(train_data=fingerprint(train_data), validation_data=fingerprint(validation_data))
    completed = restore_training(args.resume, optimizer, settings, device) if args.resume else 0
    if args.steps <= completed:
        raise ValueError("--steps must exceed the checkpoint's completed step count")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | trainable parameters: {parameters:,} / {parameters:,}", flush=True)
    print(f"Packed sequences: train={len(train_data['input_ids'])}, validation={len(validation_data['input_ids'])}", flush=True)
    print(f"Initial validation loss: {evaluate(model, validation_data, args.batch_size, device):.4f}", flush=True)
    for step in range(completed, args.steps):
        # Fresh mini-batches; the deterministic cursor continues across resumes.
        batch = training_batch(train_data, step, args.batch_size, args.seed, device)
        loss, tokens, grad_norm = train_step(model, batch, optimizer, args.max_grad_norm)
        print(f"Step {step + 1}: train_loss={loss:.4f}, target_tokens={tokens}, grad_norm={grad_norm:.4f}", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            validation_loss = evaluate(model, validation_data, args.batch_size, device)
            checkpoint = save_checkpoint(model, tokenizer, optimizer, step + 1, settings, args.output)
            print(f"Validation loss={validation_loss:.4f} | saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
