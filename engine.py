"""Single-device training primitives; no distributed synchronization is hidden here."""

import os
from pathlib import Path
import shutil
import tempfile

import torch
from torch.nn import functional as F


# One local optimizer step. A future outer optimizer belongs outside this function.
def train_step(model, batch, optimizer, max_grad_norm):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # The HF model returns vocabulary scores [batch, sequence, vocabulary].
    logits = model(**batch, use_cache=False).logits
    loss_sum, count = next_token_loss(logits, batch)
    loss = loss_sum / count
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite training loss")
    loss.backward()  # Builds gradients, not updated weights.
    # All unique parameter tensors must participate in full fine-tuning.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is None:
            raise RuntimeError(f"Missing gradient: {name}")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm, error_if_nonfinite=True,
    )
    optimizer.step()  # AdamW updates the parameters using their gradients.
    return loss.item(), count, grad_norm.item()


# Measure mean loss per valid target token, without creating a backward graph.
@torch.no_grad()
def evaluate(model, data, batch_size, device):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        for start in range(0, len(data["input_ids"]), batch_size):
            batch = {name: tensor[start:start + batch_size].to(device)
                     for name, tensor in data.items()}
            logits = model(**batch, use_cache=False).logits
            loss_sum, count = next_token_loss(logits, batch)
            total_loss += loss_sum.item()
            total_tokens += count
    finally:
        model.train(was_training)
    return total_loss / total_tokens


# Save HF-compatible weights/tokenizer plus optimizer, RNG, and data cursor state.
def save_checkpoint(model, tokenizer, optimizer, step, settings, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"step-{step:06d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}; choose another --output")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root))
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        state = {"step": step, "settings": settings, "optimizer": optimizer.state_dict(),
                 "cpu_rng": torch.get_rng_state()}
        device = next(model.parameters()).device
        if device.type == "cuda":
            state["device_rng"] = torch.cuda.get_rng_state(device)
        elif device.type == "mps":
            state["device_rng"] = torch.mps.get_rng_state()
        state["device_type"] = device.type
        torch.save(state, temporary / "training_state.pt")
        os.rename(temporary, target)  # Publish only after all files are written.
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


# Restore optimizer history and RNG after model/data construction has finished.
def restore_training(checkpoint, optimizer, settings, device):
    state = torch.load(Path(checkpoint) / "training_state.pt", map_location="cpu", weights_only=True)
    if state["settings"] != settings:
        raise ValueError("Resume settings/data changed; use the same data and training settings")
    if state["device_type"] != device.type:
        raise ValueError("Resume on the same device type; use --model CHECKPOINT for weights-only loading")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["device_rng"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["device_rng"])
    return state["step"]


# Shift exactly once: scores at position t predict the token at position t+1.
def next_token_loss(logits, batch):
    labels = next_token_labels(batch)
    count = int((labels != -100).sum().item())
    if count == 0:
        raise ValueError("Batch has no valid next-token targets")
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    return loss_sum, count


# Label for position t is the token at t+1; -100 marks positions with no valid target.
# Masking is by POSITION so a genuine EOS target still trains even when pad == EOS.
def next_token_labels(batch):
    labels = torch.full_like(batch["input_ids"], -100)
    labels[:, :-1] = batch["input_ids"][:, 1:]
    labels[:, :-1][batch["attention_mask"][:, 1:] == 0] = -100
    return labels


# Resolve one device; CPU is the CLI default to avoid unintentional GPU execution.
def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is unavailable")
    return torch.device("cuda:0" if requested == "cuda" else requested)
