"""Generate text from the original model or a full fine-tuned checkpoint."""

import argparse
import os
from pathlib import Path

import torch

from engine import choose_device
from model import DEFAULT_MODEL, load_model


# Read generation settings and load one model on one device.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or saved checkpoint directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", help="UTF-8 file with one prompt per line (batched generation)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parallel = parser.add_argument_group("parallelism (launch with torchrun; dp*tp*pp processes)")
    parallel.add_argument("--dp", type=int, default=1, help="Data-parallel replicas (prompts are split among them)")
    parallel.add_argument("--tp", type=int, default=1, help="Tensor-parallel ways (must divide the head counts)")
    parallel.add_argument("--pp", type=int, default=1, help="Pipeline stages")
    parallel.add_argument("--microbatches", type=int, help="Pipeline microbatches (default: --pp)")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("Require positive token count/temperature and 0 < top-p <= 1")
    if bool(args.prompt) == bool(args.prompt_file):
        parser.error("Provide exactly one of --prompt or --prompt-file")
    prompts = [args.prompt] if args.prompt else [
        line for line in Path(args.prompt_file).read_text().splitlines() if line.strip()]
    if not prompts or not all(prompt.strip() for prompt in prompts):
        parser.error("Prompt must contain text")
    if min(args.dp, args.tp, args.pp) < 1 or (args.microbatches is not None and args.microbatches < 1):
        parser.error("--dp, --tp, --pp and --microbatches must be positive")
    world = int(os.environ.get("WORLD_SIZE", 1))
    if args.dp * args.tp * args.pp > 1 or world > 1:
        if args.device == "mps":
            parser.error("Parallel execution supports --device cuda or cpu")
        if args.dp * args.tp * args.pp != world:
            parser.error(f"dp*tp*pp = {args.dp * args.tp * args.pp} but torchrun started {world} "
                         "processes; launch with: torchrun --standalone --nproc-per-node N inference.py ...")
        run_parallel(args, prompts)
        return
    if len(prompts) != 1:
        parser.error("Several prompts need a parallel launch (see --dp / --pp), or pass one --prompt")
    args.prompt = prompts[0]
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    model, tokenizer = load_model(args.model, device)
    print(generate_text(model, tokenizer, args.prompt, device, args.max_new_tokens,
                        args.sample, args.temperature, args.top_p))


# Parallel generation: every rank runs this under torchrun and rank 0 prints the results.
def run_parallel(args, prompts):
    from transformers import AutoConfig

    from distributed import init_parallel, rank_zero_first, run_or_exit, validate_layout
    from model import load_tokenizer
    from parallel_generate import generate_distributed
    from parallel_model import create_stage, resolve_source

    validate_layout(AutoConfig.from_pretrained(args.model), args.dp, args.tp, args.pp,
                    int(os.environ["WORLD_SIZE"]))
    ctx = init_parallel(args.dp, args.tp, args.pp, args.device)

    def work():
        rank_zero_first(ctx, lambda: resolve_source(args.model))
        tokenizer = load_tokenizer(args.model)
        # Logits are gathered (not split over the vocabulary) so every rank can choose tokens.
        model = create_stage(args.model, ctx, gather_logits=True, training=False, whole=args.pp == 1)
        texts = generate_distributed(
            model, tokenizer, prompts, ctx, args.max_new_tokens, args.sample, args.temperature,
            args.top_p, args.seed, args.microbatches or args.pp)
        if ctx.is_main:
            print("\n---\n".join(texts), flush=True)

    run_or_exit(work)


# Autoregressive inference: repeatedly predict a token and append it to the input.
@torch.inference_mode()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=64,
                  sample=False, temperature=0.8, top_p=0.9):
    model.eval()  # Disables training behavior such as dropout.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if inputs["input_ids"].shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generated tokens exceeds the model's context window")
    options = {"temperature": temperature, "top_p": top_p} if sample else {}
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=sample,
        use_cache=True, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, **options,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    main()
