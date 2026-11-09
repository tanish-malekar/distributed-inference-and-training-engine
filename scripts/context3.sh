#!/usr/bin/env bash
# Edit on the Mac; copy source and run remotely only after checking/approval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="tmalekar@context3.cs.stonybrook.edu"
REMOTE_DIR="/data01/tmalekar/distributed-inference-and-training-engine"
BASE_PYTHON="/data01/tmalekar/conda-envs/bb/bin/python"
CONTROL_PATH="${CONTEXT3_CONTROL_PATH:-/tmp/dite-%C}"
SSH_OPTIONS=(-p 130 -J tmalekar@context.cs.stonybrook.edu:130
  -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -o ControlMaster=auto
  -o "ControlPath=$CONTROL_PATH" -o ControlPersist=120)

remote() {
  ssh -n "${SSH_OPTIONS[@]}" "$HOST" "$1"
}

# Every remote download, temporary file, and output stays under /data01.
REMOTE_ENV="cd '$REMOTE_DIR'
  mkdir -p .cache tmp outputs
  export XDG_CACHE_HOME='$REMOTE_DIR/.cache'
  export HF_HOME='$REMOTE_DIR/.cache/huggingface'
  export TORCH_HOME='$REMOTE_DIR/.cache/torch'
  export TRITON_CACHE_DIR='$REMOTE_DIR/.cache/triton'
  export CUDA_CACHE_PATH='$REMOTE_DIR/.cache/cuda'
  export PIP_CACHE_DIR='$REMOTE_DIR/.cache/pip'
  export PYTHONPYCACHEPREFIX='$REMOTE_DIR/.cache/pycache'
  export TMPDIR='$REMOTE_DIR/tmp'"

sync_source() {
  remote "mkdir -p '$REMOTE_DIR'"
  local transport
  printf -v transport '%q ' ssh "${SSH_OPTIONS[@]}"
  rsync -av --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' \
    --exclude='.cache/' --exclude='tmp/' --exclude='outputs/' \
    -e "$transport" "$ROOT/" "$HOST:$REMOTE_DIR/"
}

prepare() {
  # Install in the project's venv, reusing compatible base packages read-only.
  remote "set -eu
    $REMOTE_ENV
    export CUDA_VISIBLE_DEVICES=''
    if [ ! -x .venv/bin/python ]; then
      '$BASE_PYTHON' -m venv --system-site-packages .venv
    fi
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'"
}

action="${1:-check}"
case "$action" in
  check) remote "hostname; nvidia-smi" ;;
  sync) sync_source ;;
  setup) sync_source; prepare ;;
  run|infer)
    gpu="${2:-}"
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
      echo 'Supply a GPU index. See CONTEXT3.md.' >&2; exit 2
    fi
    if [[ "$action" == run ]]; then
      steps="${3:-1}"
      if ! [[ "$steps" =~ ^[1-9][0-9]*$ ]]; then
        echo 'Step count must be positive.' >&2; exit 2
      fi
      if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi
      command=(.venv/bin/python -u train.py "$@" --steps "$steps" --device cuda)
    else
      shift 2
      command=(.venv/bin/python -u inference.py "$@" --device cuda)
    fi
    sync_source
    prepare
    remote "cd '$REMOTE_DIR' && .venv/bin/python scripts/check_gpu.py '$gpu'"
    printf '\nRequested job: '; printf '%q ' "${command[@]}"
    printf '\nApprove this Context3 GPU %s job? Type RUN GPU %s: ' "$gpu" "$gpu"
    IFS= read -r confirmation
    if [[ "$confirmation" != "RUN GPU $gpu" ]]; then
      echo 'Cancelled. No GPU job started.'; exit 1
    fi
    # Quote every argument; prompts and paths must not become remote shell code.
    printf -v job '%q ' "${command[@]}"
    remote "set -eu
      $REMOTE_ENV
      .venv/bin/python scripts/check_gpu.py '$gpu'
      export CUDA_VISIBLE_DEVICES='$gpu'
      exec $job"
    ;;
  *) echo 'Usage: context3.sh {check|sync|setup|run GPU STEPS [OPTIONS...]|infer GPU [OPTIONS...]}' >&2; exit 2 ;;
esac
