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
```

Every training **and inference** GPU job requires fresh approval. The helper checks
utilization, memory, and processes, displays the exact job, asks for `RUN GPU 0`
(or the chosen GPU), then rechecks immediately before launching. It rejects any
listed process, nonzero utilization, or memory beyond 10 MiB driver baseline.
This is a point-in-time availability check, not an exclusive scheduler reservation.
Only the selected GPU is visible. Do not run another GPU job without fresh approval.

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
