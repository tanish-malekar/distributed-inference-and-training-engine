"""Generate text from the original model or a full fine-tuned checkpoint."""

import argparse

import torch

from engine import choose_device
from model import DEFAULT_MODEL, load_model


# Read generation settings and load one model on one device.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or saved checkpoint directory")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("Require positive token count/temperature and 0 < top-p <= 1")
    if not args.prompt.strip():
        parser.error("Prompt must contain text")
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    model, tokenizer = load_model(args.model, device)
    print(generate_text(model, tokenizer, args.prompt, device, args.max_new_tokens,
                        args.sample, args.temperature, args.top_p))


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
