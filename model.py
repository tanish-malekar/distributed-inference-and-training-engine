"""Load a real Hugging Face causal language model without rewriting its layers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"


# Load architecture, pretrained weights, and the matching tokenizer on one device.
def load_model(source: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.eos_token_id is None:
        raise ValueError("This baseline requires a tokenizer with an EOS token")
    # EOS can also fill padding positions. The attention mask, NOT token identity,
    # distinguishes real EOS tokens from padding when calculating the loss.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, attn_implementation="eager",
    ).to(device)
    # Every parameter trains. No LoRA, quantization, device_map, or sharding.
    model.requires_grad_(True)
    model.config.pad_token_id = tokenizer.pad_token_id
    # Training uses whole sequences. Inference enables the KV cache separately.
    model.config.use_cache = False
    return model, tokenizer
