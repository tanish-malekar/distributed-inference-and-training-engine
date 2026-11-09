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
  run|infer|smoke|nccl)
    gpus="${2:-}"
    # One index (0) or a comma-separated list (0,1,2,3). Only these GPUs are made visible.
    if ! [[ "$gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
      echo 'Supply a GPU index or comma-separated list, e.g. 0 or 0,1. See CONTEXT3.md.' >&2; exit 2
    fi
    IFS=, read -r -a gpu_list <<< "$gpus"
    if [[ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l | tr -d ' ')" != "${#gpu_list[@]}" ]]; then
      echo 'A GPU index appears more than once.' >&2; exit 2
    fi
    n="${#gpu_list[@]}"
    if [[ "$action" == run ]]; then
      steps="${3:-1}"
      if ! [[ "$steps" =~ ^[1-9][0-9]*$ ]]; then
        echo 'Step count must be positive.' >&2; exit 2
      fi
      if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi
      script=train.py; extra=(--steps "$steps")
    elif [[ "$action" == nccl ]]; then
      shift 2
      script=scripts/nccl_check.py; extra=()
      set -- --dp "$n"   # only the process count matters for this check
    elif [[ "$action" == smoke ]]; then
      shift 2
      script=scripts/parallel_smoke_test.py; extra=()
    else
      shift 2
      script=inference.py; extra=()
    fi
    # dp * tp * pp (each defaults to 1) must equal the number of approved GPUs.
    dp=1; tp=1; pp=1; args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do
      case "${args[i]}" in
        --dp) dp="${args[i+1]:-}" ;; --dp=*) dp="${args[i]#--dp=}" ;;
        --tp) tp="${args[i+1]:-}" ;; --tp=*) tp="${args[i]#--tp=}" ;;
        --pp) pp="${args[i+1]:-}" ;; --pp=*) pp="${args[i]#--pp=}" ;;
      esac
    done
    for degree in "$dp" "$tp" "$pp"; do
      if ! [[ "$degree" =~ ^[1-9][0-9]*$ ]]; then
        echo '--dp, --tp and --pp must be positive integers.' >&2; exit 2
      fi
    done
    if (( dp * tp * pp != n )); then
      echo "dp*tp*pp = $((dp * tp * pp)) but $n GPU(s) were listed; they must match." >&2; exit 2
    fi
    if (( n == 1 )); then
      command=(.venv/bin/python -u "$script" "$@" ${extra[@]+"${extra[@]}"} --device cuda)
      phrase="RUN GPU $gpus"
    else
      # One process per approved GPU, single node; torch.distributed.run is torchrun's module form.
      command=(.venv/bin/python -u -m torch.distributed.run --standalone --nproc-per-node "$n"
        "$script" "$@" ${extra[@]+"${extra[@]}"} --device cuda)
      phrase="RUN GPUS $gpus"
    fi
    # Optional: CONTEXT3_TIMEOUT=SECONDS stops a hung job; CONTEXT3_ENV="NCCL_DEBUG=INFO ..." sets
    # remote environment variables (names and simple values only, so nothing can inject shell code).
    if [[ -n "${CONTEXT3_TIMEOUT:-}" ]]; then
      if ! [[ "$CONTEXT3_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
        echo 'CONTEXT3_TIMEOUT must be a positive number of seconds.' >&2; exit 2
      fi
      command=(timeout -k 10 "$CONTEXT3_TIMEOUT" "${command[@]}")
    fi
    job_env=""
    for assignment in ${CONTEXT3_ENV:-}; do
      if ! [[ "$assignment" =~ ^[A-Z_][A-Z0-9_]*=[A-Za-z0-9_.,:/-]*$ ]]; then
        echo "Rejected CONTEXT3_ENV entry: $assignment" >&2; exit 2
      fi
      job_env+="export $assignment; "
    done
    # Direct GPU-to-GPU (P2P) transfers hang on Context3 (an NCCL all_reduce between GPUs 0 and 2
    # never completes); staging through host memory works. Override with CONTEXT3_ENV=NCCL_P2P_DISABLE=0.
    if (( n > 1 )) && [[ " ${CONTEXT3_ENV:-} " != *" NCCL_P2P_DISABLE="* ]]; then
      job_env+="export NCCL_P2P_DISABLE=1; "
    fi
    sync_source
    prepare
    remote "cd '$REMOTE_DIR' && .venv/bin/python scripts/check_gpu.py '$gpus'"
    printf '\nRequested job on GPU(s) %s' "$gpus"
    [[ -n "$job_env" ]] && printf ' with environment: %s' "${job_env//export /}"
    printf ': '; printf '%q ' "${command[@]}"
    printf '\nApprove this Context3 GPU job? Type %s: ' "$phrase"
    IFS= read -r confirmation
    if [[ "$confirmation" != "$phrase" ]]; then
      echo 'Cancelled. No GPU job started.'; exit 1
    fi
    # Quote every argument; prompts and paths must not become remote shell code.
    printf -v job '%q ' "${command[@]}"
    remote "set -eu
      $REMOTE_ENV
      .venv/bin/python scripts/check_gpu.py '$gpus'
      export CUDA_VISIBLE_DEVICES='$gpus'
      $job_env
      exec $job"
    ;;
  *) echo 'Usage: context3.sh {check|sync|setup|run GPUS STEPS [OPTIONS...]|infer GPUS [OPTIONS...]|smoke GPUS --output DIR [OPTIONS...]|nccl GPUS}' >&2; exit 2 ;;
esac
