# Context3 rules

- Keep the editable source checkout on the user's Mac.
- Context3 is reached through context.cs.stonybrook.edu. Both SSH ports are 130.
- Before EVERY GPU execution on Context3, inspect the utilization, memory usage, and
  running processes of EACH selected GPU, then obtain the user's explicit approval
  for those GPUs and that run. Recheck every selected GPU immediately before launching.
  Earlier approval is not blanket approval for later runs. Read-only nvidia-smi checks
  are allowed.
- Use only the GPUs the user approved for that run, all on the single Context3 node
  (data, tensor and pipeline parallel runs need one GPU per process). Never expand
  to additional GPUs, and never launch multi-node jobs, without fresh approval.
- Store large files and environments under /data01/tmalekar, never /home/tmalekar.
  This includes datasets, checkpoints, dependency downloads, and caches.
- Never put passwords or private keys in source code, configuration, or docs.

# Code organization (this repository only)

- Order classes and functions from high-level entry points to the lower-level helpers they use,
  so readers can understand a file from the top down.
- Keep configuration/data classes near the top. In executable files, put main()
  before run() and supporting helpers; keep the __main__ invocation at the bottom.
- model.py loads the real HF model; engine.py holds local steps, evaluation,
  checkpointing, and device helpers. Keep data preparation separate in data.py.
- engine.py stays the single-device reference that the parallel path must match. The
  multi-process path lives in distributed.py (process mesh), parallel_model.py (stage,
  tensor/data parallel), parallel_engine.py (pipeline step), parallel_checkpoint.py,
  parallel_generate.py and parallel_train.py. Prefer PyTorch's built-in parallelism
  (DTensor, FSDP2, pipelining, DCP) over custom communication code.
- Preserve comments with their classes and functions when reordering. Follow this convention
  for future code in this repository, while respecting Python's definition-time
  dependencies (for example, a base class must exist before its subclass).
