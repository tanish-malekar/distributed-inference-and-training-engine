# Distributed Inference and Training Engine

Full fine-tuning and autoregressive inference for **HuggingFaceTB/SmolLM2-135M**,
using its existing Hugging Face Llama implementation and pretrained tokenizer/weights.
All model parameters train on one device in one process. Start with `train.py`.

## Install and run locally

Python 3.11+ is recommended. Commands below run from this project directory.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train.py --device cpu --steps 1 --output outputs/first-run
.venv/bin/python inference.py --device cpu --prompt "The purpose of machine learning is"
.venv/bin/python inference.py --model outputs/first-run/step-000001 --prompt "The purpose of machine learning is"
```

The first run downloads the model and WikiText dataset. The default device is CPU;
use `--device mps` for the Mac GPU. For Context3, use the guarded helper described in
[CONTEXT3.md](CONTEXT3.md), including approval before each GPU job. Full fine-tuning
uses considerably more memory than the model weights alone (gradients, AdamW state,
and activations also occupy memory). CPU checks can be slow.

## Training

```bash
.venv/bin/python train.py --steps 100 --batch-size 2 --sequence-length 128 \
  --learning-rate 5e-5 --eval-every 25 --output outputs/experiment-1
```

Defaults use the first 1,000 training rows and 200 held-out validation rows from
`Salesforce/wikitext`, configuration `wikitext-2-raw-v1`. These are small learning
subsets, not a benchmark. Increase `--train-rows` and `--validation-rows` for larger
runs. Empty rows are skipped; text documents are separated by EOS and packed into
fixed-length sequences. Each epoch reshuffles the sequences. A final short sequence
is right-padded, and padding targets are excluded from loss by their attention mask.
Packing allows attention across document boundaries; this baseline does not implement
isolated document attention. An epoch's final mini-batch may be smaller.

For your own data, provide **both** separate UTF-8 files, one document per line:

```bash
.venv/bin/python train.py --train-file train.txt --validation-file validation.txt \
  --steps 20 --output outputs/my-data
```

The row limits apply to these files too. Keep validation text separate from training.

The local step is deliberately explicit in `engine.py`:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(**batch, use_cache=False).logits
loss_sum, valid_tokens = next_token_loss(logits, batch)
loss = loss_sum / valid_tokens
loss.backward()
# Check gradient presence and clip the global gradient norm.
optimizer.step()
```

Scores at token position `t` predict the ID at `t+1`. We shift once in our loss
function; we do not also pass shifted labels to Hugging Face. Real EOS tokens still
contribute to loss even when EOS is used as padding. Validation sums losses and counts
tokens across batches, so its mean weights every valid target token equally.

Every unique parameter is trainable. AdamW uses constant learning rate and weight
decay; global gradient clipping defaults to 1.0. The printed norm is the norm before
clipping. FP32 and eager attention keep this baseline straightforward: no mixed
precision, accumulation, scheduler, LoRA, quantization, or distributed execution yet.
A short run is not expected to produce consistently decreasing validation loss.

## Checkpoint and resume

At evaluation intervals and the final step, save `outputs/.../step-NNNNNN/`:

- HF-compatible model configuration/weights and tokenizer, loadable by inference.
- `training_state.pt`: AdamW state, completed steps, RNG, data fingerprints, and settings.

```bash
.venv/bin/python train.py --resume outputs/experiment-1/step-000100 \
  --steps 150 --output outputs/experiment-1
```

`--steps` is the total target count: this adds 50 steps. Use the same batch size,
sequence length, seed, optimizer settings, text, and row limits as the original run.
The data cursor resumes from the completed step. Resume requires the same device type;
exact bitwise results across different hardware/software are not guaranteed. A weights-only
restart uses `--model CHECKPOINT` instead of `--resume`, with a new optimizer and cursor.
Existing checkpoint directories are never overwritten; choose a fresh output for a new run.
Each saved checkpoint includes full weights and AdamW state, so storage grows with saves.

## Inference

```bash
.venv/bin/python inference.py --model outputs/experiment-1/step-000150 \
  --prompt "Distributed training is" --max-new-tokens 64
.venv/bin/python inference.py --prompt "Once upon a time" --sample --temperature 0.8 --top-p 0.9
```

Default decoding is greedy. Generation uses eval/inference mode, reuses the KV cache,
and stops on EOS or the requested token limit. The base model completes text; it is
not an instruction/chat model. The decoded output includes the prompt.

## Reading order and future extensions

- `train.py`: high-level training orchestration and command-line settings.
- `model.py`: load the existing HF architecture, checkpoint, and tokenizer.
- `data.py`: real text, packing, mini-batches, and resume fingerprints.
- `engine.py`: backward/update, evaluation, checkpointing, and device selection.
- `inference.py`: text generation using the same saved model format.
- `scripts/`: guarded Context3 execution and read-only GPU availability checks.

The local optimizer step is separate from orchestration. Future DiLoCo outer updates
can live around several local steps. Replica communication/compression and pipeline
stage communication will require different integration points. Backward communication
overlap will use runtime hooks rather than a call after backward. Different LoRA ranks
will require an aggregation policy. Those algorithms are not implemented here.
We retain HF's module structure, so future partitioning can access its existing layers;
SmolLM2 has 9 query heads and 3 KV heads: a future tensor-parallel degree must
respect those dimensions; splitting it across two TP workers is not automatic.
Using another `--model` is possible for compatible causal LMs but is not a promise of
support for arbitrary architectures or remote custom code.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests run offline on CPU with a tiny instance of the **real HF Llama architecture**
and a local tokenizer. They exercise full backward/updates, loss alignment/padding,
checkpoint reload, exact CPU continuation, and generation. They do not download
pretrained weights or demonstrate fine-tuning quality of SmolLM2.

References: [model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
[WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
[HF causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling).

Verified locally with Python 3.12, PyTorch 2.15.0, Transformers 5.22.2, and Datasets
4.8.5. A separate CPU smoke run loaded the actual pretrained SmolLM2 checkpoint,
trained one step on WikiText (10 source training rows, 3 validation rows, batch 1,
sequence length 16), saved it, resumed to step 2, and generated text from the saved
weights. This verifies execution, not model quality. The ignored test checkpoints
are under `.cache/smoke/pretrained/`; no Context3 GPU run was performed.
