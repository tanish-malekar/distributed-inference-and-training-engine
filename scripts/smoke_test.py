"""Short end-to-end full fine-tuning, checkpoint resume, and generation test.

On Context3, check the GPU and obtain approval before launching this script.
"""

import argparse
from pathlib import Path
import sys

# Allow launching from the project root as: python scripts/smoke_test.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine import choose_device
from inference import generate_text
from model import DEFAULT_MODEL, load_model
from train import run


# Keep the selected device and output directory explicit for each test job.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Choose a fresh output directory")
    run_test(args.device, args.output)


# Exercise the actual pretrained model and both saved-model loading paths.
def run_test(requested_device, output):
    device = choose_device(requested_device)
    settings = argparse.Namespace(
        model=DEFAULT_MODEL, device=requested_device, steps=2, batch_size=2,
        sequence_length=128, learning_rate=5e-5, weight_decay=0.01,
        max_grad_norm=1.0, eval_every=25, train_rows=100, validation_rows=20,
        train_file=None, validation_file=None, seed=42, output=output, resume=None,
    )
    print("TEST 1: two full fine-tuning steps and checkpoint saving", flush=True)
    run(settings)
    settings.resume = str(Path(output) / "step-000002")
    settings.steps = 3
    print("TEST 2: restore optimizer/data cursor and continue to step three", flush=True)
    run(settings)
    print("TEST 3: load the fine-tuned checkpoint for generation", flush=True)
    model, tokenizer = load_model(str(Path(output) / "step-000003"), device)
    text = generate_text(model, tokenizer, "The purpose of machine learning is", device, 24)
    if not text.strip():
        raise RuntimeError("Generation produced empty text")
    print(text, flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print(f"Peak GPU allocated memory: {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)
    print("PASS: training, validation, checkpoint save/resume, and inference", flush=True)


if __name__ == "__main__":
    main()
