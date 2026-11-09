# Edit locally in VS Code; execute on Context3

Source: `/Users/tanish/Desktop/Repos/Distributed Inference and Training Engine`.
Remote copy: `/data01/tmalekar/distributed-inference-and-training-engine`.
All remote environments, downloads, caches, temporary files, datasets, and checkpoints
stay under `/data01/tmalekar`, never `/home/tmalekar`.

Open the local folder in VS Code. Save changes, then use **Tasks: Run Task**:

1. **Context3: Check GPUs (read-only)**.
2. **Context3: Set up and sync (no GPU)** installs dependencies in the project's venv.
3. **Context3: Fine-tune one step (confirmation required)**.
4. **Context3: Generate text (confirmation required)** accepts a model/checkpoint and prompt.
5. **Context3: Parallel smoke test (confirmation required)** trains, checkpoints, resumes
   and generates under the chosen layout.

The GPU prompt takes one index (`0`) or a comma-separated list (`0,1`); the data, tensor
and pipeline degrees default to 1, and `dp * tp * pp` must equal the number of GPUs listed.

Tasks execute remotely and display output in the local terminal. The Python Run button
still executes locally; these tasks do not provide remote breakpoint debugging.

Equivalent terminal commands:

```bash
bash scripts/context3.sh check
bash scripts/context3.sh setup
bash scripts/context3.sh run 0 1 --output outputs/first-run
bash scripts/context3.sh run 0 100 --output outputs/experiment-1
bash scripts/context3.sh run 0 150 --resume outputs/experiment-1/step-000100 --output outputs/experiment-1
bash scripts/context3.sh infer 0 --model outputs/first-run/step-000001 --prompt "Machine learning is"
# Several GPUs: one process per GPU, dp*tp*pp = number of GPUs listed
bash scripts/context3.sh run 0,1 100 --dp 2 --output outputs/dp2
bash scripts/context3.sh run 0,1,2 100 --tp 3 --output outputs/tp3
bash scripts/context3.sh run 0,1,2,3 100 --dp 2 --pp 2 --dp-shard --output outputs/dp2-pp2
bash scripts/context3.sh infer 0,1 --model outputs/dp2/step-000100 --pp 2 --prompt "Machine learning is"
bash scripts/context3.sh smoke 0,1 --pp 2 --output outputs/smoke-pp2
```

Every training **and inference** GPU job requires fresh approval. The helper checks
utilization, memory, and processes of **each** listed GPU, displays the exact job, asks you
to type `RUN GPU 0` (one GPU) or `RUN GPUS 0,1` (several, exactly as listed), then rechecks
every GPU immediately before launching. It rejects any listed process, nonzero utilization,
or memory beyond 10 MiB driver baseline on any of them. This is a point-in-time availability
check, not an exclusive scheduler reservation. Only the listed GPUs are visible, on this one
node (no multi-node jobs). Do not run another GPU job without fresh approval.

**Known cluster issue:** direct GPU-to-GPU (P2P) transfers hang on Context3: a bare NCCL
`all_reduce` between GPUs 0 and 2 never completes, while `NCCL_P2P_DISABLE=1` passes
all_reduce, barrier and send/recv. The helper therefore sets `NCCL_P2P_DISABLE=1` for every
multi-GPU job (override: `CONTEXT3_ENV=NCCL_P2P_DISABLE=0`). `bash scripts/context3.sh nccl 0,2`
runs that minimal NCCL check (also with `CONTEXT3_TIMEOUT=90`, `CONTEXT3_ENV="NCCL_DEBUG=INFO"`).

Multi-GPU jobs start through `python -m torch.distributed.run --standalone`, one process
per GPU, with NCCL. SmolLM2 supports `--tp 1` or `--tp 3` (9 query / 3 KV heads), so
tensor parallelism uses 3 GPUs per tensor group; `--pp` may be up to 30. See
[README.md](README.md#parallelism-data-tensor-and-pipeline). The multi-GPU path has been
verified on 2-3 GPUs (dp2, dp3, pp2, pp3, tp3, GPipe, FSDP, resume across layouts); four-GPU
and larger layouts have only run on CPU. To validate a new layout, run `smoke` on the smallest one first (for
example 2 GPUs with `--dp 2`), then compare its step losses with a one-GPU run of identical
settings: `run 0 3 --batch-size B --sequence-length 128 --train-rows 100 --validation-rows 20
--eval-every 25 --output ...` where `B = 2 * dp * microbatches` (`microbatches` defaults to `--pp`).

The first training run downloads SmolLM2-135M and WikiText. It performs full FP32
fine-tuning. A one-step run still includes validation and a full resumable checkpoint.
Each checkpoint can occupy around 1.6 GB for FP32 weights and AdamW state; save fewer
checkpoints using `--eval-every` if necessary. Later runs reuse the cached downloads.

SSH uses your account via Context, both on port 130:

```bash
ssh -J tmalekar@context.cs.stonybrook.edu:130 -p 130 tmalekar@context3.cs.stonybrook.edu
```

Enter your password only when SSH asks. No credentials are stored in this repo.
The helper requires trusted host keys. It reuses compatible packages from
`/data01/tmalekar/conda-envs/bb` read-only in a separate project venv, and installs
additional dependencies into that venv. It does not modify the base environment.

Sync excludes Git, local venv, caches, temporary files, and outputs; it never deletes
remote files. Relative data/checkpoint paths passed to the helper refer to the remote
project directory. To use local custom text data, put it in a source subdirectory
(e.g. `text/`) before syncing. Large datasets should be fetched directly under /data01.
