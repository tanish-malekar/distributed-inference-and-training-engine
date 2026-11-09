# Distributed Inference and Training Engine

Full fine-tuning and autoregressive inference for **HuggingFaceTB/SmolLM2-135M**,
using its existing Hugging Face Llama implementation and pretrained tokenizer/weights.
All model parameters train. By default everything runs on one device in one process; the
same commands also scale out with **data, tensor and pipeline parallelism** (see
[Parallelism](#parallelism-data-tensor-and-pipeline)). Start with `train.py`.

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

## Parallelism: data, tensor and pipeline

Add `--dp`, `--tp` and `--pp` to `train.py` or `inference.py` and launch **one process per
rank**, `dp * tp * pp` in total. The three degrees combine freely on a (pp, dp, tp) mesh
with tensor-parallel peers on adjacent ranks. `python -m torch.distributed.run` is
`torchrun`'s module form (the venv's `torchrun` launcher script may carry a stale path).

```bash
# 2 data-parallel replicas x 3-way tensor parallel = 6 processes (CPU/gloo here)
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 6 train.py \
  --device cpu --dp 2 --tp 3 --steps 100 --output outputs/dp2-tp3
# 2 pipeline stages, 4 microbatches, optimizer/gradient state sharded over 2 replicas (FSDP)
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 4 train.py \
  --device cpu --dp 2 --pp 2 --microbatches 4 --dp-shard --steps 100 --output outputs/dp2-pp2
# generation: prompts are split over replicas, layers over stages, weights over tensor ranks
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 6 inference.py \
  --device cpu --model outputs/dp2-tp3/step-000100 --tp 3 --pp 2 --prompt-file prompts.txt
```

| Axis | What is split | Built on |
|---|---|---|
| `--dp` | the batch; each replica holds all weights (`--dp-shard` shards weights, gradients and AdamW state) | FSDP2 `replicate` / `fully_shard` |
| `--tp` | every attention head group and MLP column/row, plus the vocabulary (embedding, output layer and loss) | DTensor `parallelize_module`, `loss_parallel` |
| `--pp` | contiguous blocks of decoder layers (`--pp-schedule 1f1b` or `gpipe`, `--microbatches`, default `--pp`) | `torch.distributed.pipelining` |

**Constraints.** `dp*tp*pp` must equal the process count. `--tp` must divide the query heads,
KV heads, MLP width and vocabulary: for SmolLM2 (9 query / 3 KV heads) that means
`--tp 1` or `--tp 3`; `--pp` can be anything up to the 30 layers. 1F1B needs
`--microbatches >= --pp`. The model config must not set `pad_token_id` when `--tp > 1`
(DTensor embeddings lack `padding_idx`). CUDA (NCCL) and CPU (gloo) are supported, MPS is not.

**The result is the single-device result.** Loss is the sum of token losses divided by the
global valid-token count, so every layout gives the same loss, gradient and update as
`engine.train_step` (up to floating-point summation order). A short last batch, and a batch
that does not divide `dp * microbatches`, is padded to a fixed size with fully masked rows
that contribute exactly zero, because pipeline stages fix their buffer shapes on the first
step. Gradient clipping uses the true global norm across stages, tensor shards and replicas.
With `--pp > 1` the tied embedding/output matrix lives on the first and last stage; their
gradients are summed so both copies stay identical, and it is counted once in the norm and in
the printed parameter count.

**Checkpoints are layout-independent.** `step-N/` holds ordinary Hugging Face files
(consolidated from every rank, loadable by `inference.py`, `load_model` and `--model`) plus
`optimizer/` (a `torch.distributed.checkpoint` of the AdamW state, resharded on load),
`training_state.pt` and per-rank RNG. `--resume` therefore works under a **different**
`--dp/--tp/--pp` than the run that wrote it, from or into a single process. RNG state is
restored only when the layout is unchanged (SmolLM2 uses no dropout).

**Inference.** With `--pp 1` the complete HF model runs through HF `generate` and its KV
cache, with tensor-parallel weights and prompts dealt out over `--dp` replicas. With
`--pp > 1` a small decode loop runs each stage's layers with an HF `DynamicCache` and passes
hidden states between stages; microbatches keep stages busy and the last stage's tokens are
broadcast each step. Batched prompts are left-padded. Greedy output equals single-device
`generate`; sampling draws identically on every tensor-parallel rank.

Not implemented: sequence/context parallelism, interleaved or zero-bubble pipeline schedules,
mixed precision, multi-node jobs, and overlapping the FSDP reduce with the pipeline (each
microbatch's backward reduces its gradient). On macOS, if `--standalone` hangs with IPv6
warnings, launch with `--rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29555 --local-addr=127.0.0.1`.

## Reading order and future extensions

- `train.py`: high-level training orchestration and command-line settings.
- `model.py`: load the existing HF architecture, checkpoint, and tokenizer.
- `data.py`: real text, packing, mini-batches, and resume fingerprints.
- `engine.py`: backward/update, evaluation, checkpointing, and device selection. This
  single-device path is the reference every parallel layout is tested against.
- `inference.py`: text generation using the same saved model format.
- `distributed.py`: the (pp, dp, tp) process mesh and layout validation.
- `parallel_model.py`: one pipeline stage of the HF model with tensor/data parallelism.
- `parallel_engine.py`: pipeline schedule, global-norm clipping, distributed evaluation.
- `parallel_checkpoint.py`, `parallel_generate.py`, `parallel_train.py`: layout-independent
  checkpoints, parallel decoding, and the multi-process training driver.
- `scripts/`: guarded Context3 execution, read-only GPU availability checks, smoke tests.

The local optimizer step is separate from orchestration. Future DiLoCo outer updates
can live around several local steps, and different LoRA ranks will require an aggregation
policy; neither is implemented here. We retain HF's module structure and parameter names,
which is what lets the parallel path partition the existing layers and keep checkpoints
interchangeable. Using another `--model` is possible for compatible causal LMs but is not a promise of
support for arbitrary architectures or remote custom code.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests run offline on CPU with a tiny instance of the **real HF Llama architecture**
and a local tokenizer. They exercise full backward/updates, loss alignment/padding,
checkpoint reload, exact CPU continuation, and generation. They do not download
pretrained weights or demonstrate fine-tuning quality of SmolLM2.

`tests/test_parallel.py` additionally spawns up to 8 real processes (gloo) per test and
checks every parallel layout against the single-device run: per-step loss and gradient norm,
final weights, token-weighted validation loss, the short padded last batch, layout-changing
checkpoint resume (including into and out of a pipeline), the Hugging Face export, and
greedy/sampled generation across ranks. The whole suite takes under a minute.

References: [model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M),
[WikiText](https://huggingface.co/datasets/Salesforce/wikitext),
[HF causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling).

Verified locally with Python 3.12, PyTorch 2.15.0, Transformers 5.22.2, and Datasets
4.8.5. A separate CPU smoke run loaded the actual pretrained SmolLM2 checkpoint,
trained one step on WikiText (10 source training rows, 3 validation rows, batch 1,
sequence length 16), saved it, resumed to step 2, and generated text from the saved
weights. This verifies execution, not model quality. The ignored test checkpoints
are under `.cache/smoke/pretrained/`.

The parallel path was also run on the actual pretrained SmolLM2 on this Mac's CPU (gloo):
dp2xtp3, tp3xpp2 and dp2xpp2 with `--dp-shard` reproduced the single-device train loss,
validation loss and (to about 1e-4) gradient norm at every step of identical settings;
a checkpoint written by dp2xtp3 resumed under dp2xpp2 and matched; and tp3xpp2 generation
matched single-device greedy output.

On Context3 (RTX A6000s, NCCL, the cluster's PyTorch 2.11) each axis was run on its own through
`scripts/context3.sh smoke` (train 3 steps with a checkpoint and resume, then generate) and compared
with a one-GPU `train.py` run of identical settings: dp2, dp2 with `--dp-shard`, dp3, pp2, pp3,
GPipe with 4 microbatches, and tp3 all reproduced the baseline loss, gradient norm and validation
loss to four decimals at every step. A pipeline checkpoint resumed under tp3 and matched. Inference
gave identical text on 1 GPU, pp2, pp3 and tp3. The whole `tests/` suite (24 tests, including the
8-process dp2xtp2xpp2 case) also passes on the cluster's CPUs with PyTorch 2.11.
**Not run on GPUs:** any layout needing four or more GPUs (dp2xpp2, tpxpp, tpxdp, 3D), because
one of the four GPUs was in use by another job. Those layouts are covered by the CPU tests only.
