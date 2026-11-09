"""Real text -> token IDs -> packed sequences -> deterministic mini-batches."""

import hashlib
from pathlib import Path

import torch


# Load independent training/validation splits and prepare fixed-length sequences.
def prepare_data(tokenizer, sequence_length, train_rows, validation_rows,
                 train_file=None, validation_file=None):
    if train_file:
        train_texts = Path(train_file).read_text().splitlines()[:train_rows]
        validation_texts = Path(validation_file).read_text().splitlines()[:validation_rows]
    else:
        from datasets import load_dataset
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        train_texts = dataset["train"].select(range(min(train_rows, len(dataset["train"]))))["text"]
        validation_texts = dataset["validation"].select(
            range(min(validation_rows, len(dataset["validation"]))))["text"]
    return (pack_texts(train_texts, tokenizer, sequence_length),
            pack_texts(validation_texts, tokenizer, sequence_length))


# Join documents with EOS and pack tokens; keep a right-padded final block.
def pack_texts(texts, tokenizer, sequence_length):
    tokens = []
    for text in texts:
        if text.strip():
            tokens.extend(tokenizer.encode(text, add_special_tokens=False))
            tokens.append(tokenizer.eos_token_id)
    rows, masks = [], []
    for start in range(0, len(tokens), sequence_length):
        chunk = tokens[start:start + sequence_length]
        # One token alone cannot provide a next-token training target.
        if len(chunk) < 2:
            continue
        padding = sequence_length - len(chunk)
        rows.append(chunk + [tokenizer.pad_token_id] * padding)
        masks.append([1] * len(chunk) + [0] * padding)
    if not rows:
        raise ValueError("Not enough text for a next-token example")
    return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor(masks)}


# Select successive shuffled batches; epoch order is independent of global RNG.
def training_batch(data, step, batch_size, seed, device):
    batches_per_epoch = (len(data["input_ids"]) + batch_size - 1) // batch_size
    epoch, batch_index = divmod(step, batches_per_epoch)
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(data["input_ids"]), generator=generator)
    indices = order[batch_index * batch_size:(batch_index + 1) * batch_size]
    return {name: tensor[indices].to(device) for name, tensor in data.items()}


# Detect changes in tokenized data before resuming an optimizer/data cursor.
def fingerprint(data):
    digest = hashlib.sha256()
    for name, tensor in data.items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
