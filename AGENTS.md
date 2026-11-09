# Context3 rules

- Keep the editable source checkout on the user's Mac.
- Context3 is reached through context.cs.stonybrook.edu. Both SSH ports are 130.
- Before EVERY GPU execution on Context3, inspect the selected GPU's utilization,
  memory usage, and running processes, then obtain the user's explicit approval
  for that GPU and run. Recheck immediately before launching. Earlier approval
  is not blanket approval for later runs. Read-only nvidia-smi checks are allowed.
- Use only one approved GPU for this learning exercise. No distributed execution.
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
- Preserve comments with their classes and functions when reordering. Follow this convention
  for future code in this repository, while respecting Python's definition-time
  dependencies (for example, a base class must exist before its subclass).
