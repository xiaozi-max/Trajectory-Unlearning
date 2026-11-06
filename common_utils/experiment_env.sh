#!/usr/bin/env bash
set -o pipefail

RISK_GIRPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RISK_GIRPO_ROOT

# Load machine-local values once. Child launchers inherit the marker, so
# explicit overrides made by collect/suite scripts are not reset from .env.
if [[ -z "${RISK_GIRPO_ENV_LOADED:-}" && -f "$RISK_GIRPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$RISK_GIRPO_ROOT/.env"
    set +a
    export RISK_GIRPO_ENV_LOADED=1
fi

export HF_HOME="${HF_HOME:-$RISK_GIRPO_ROOT/work/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-risk-aware-girpo}"
export WANDB_DIR="${WANDB_DIR:-$RISK_GIRPO_ROOT/work/wandb}"
export TMPDIR="${TMPDIR:-/tmp/risk-girpo-${UID:-user}}"
mkdir -p "$HF_HOME" "$WANDB_DIR" "$TMPDIR"

progress_heartbeat() {
    local state_dir="$1" interval="${PROGRESS_INTERVAL_SECONDS:-15}"
    local started now elapsed phase gpu_ids query
    started="$(date +%s)"
    while true; do
        sleep "$interval"
        now="$(date +%s)"; elapsed=$((now - started))
        phase="$(cat "$state_dir/phase" 2>/dev/null || echo starting)"
        gpu_ids="$(cat "$state_dir/gpus" 2>/dev/null || true)"
        if [[ -n "$gpu_ids" ]]; then
            query="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F, -v ids="$gpu_ids" 'BEGIN{gsub(/,/," ",ids)} index(" " ids " "," " $1 " "){gsub(/^ +| +$/,"",$1); gsub(/^ +| +$/,"",$2); gsub(/^ +| +$/,"",$3); gsub(/^ +| +$/,"",$4); printf "GPU%s=%s/%sMB util=%s%% ",$1,$2,$3,$4}')"
        else
            query="waiting for GPU selection"
        fi
        echo "[progress] time=$(date '+%Y-%m-%d %H:%M:%S') elapsed=${elapsed}s phase=$phase $query"
    done
}

grab_gpus() {
    local count="${1:-1}" memory="${2:-30000}" util="${3:-10}" max_proc="${4:-0}"
    local ids="${GPU_IDS:-0 1 2 3 4 5 6 7}" wait_seconds="${GPU_WAIT_SECONDS:-60}"
    local selected attempt=0
    echo "[gpu-select] need=$count free_memory>=${memory}MB util<=${util}% processes<=${max_proc} candidates=[$ids]"
    while true; do
        attempt=$((attempt + 1))
        # shellcheck disable=SC2086
        selected="$(python "$RISK_GIRPO_ROOT/common_utils/get_gpu_ids.py" \
            -N "$count" -M "$memory" -U "$util" -P "$max_proc" -I $ids -v)"
        if [[ -n "$selected" ]]; then
            export CUDA_VISIBLE_DEVICES="$selected"
            echo "[gpu-select] attempt=$attempt selected=$CUDA_VISIBLE_DEVICES"
            return 0
        fi
        echo "[gpu-select] attempt=$attempt no qualifying GPU; retrying in ${wait_seconds}s" >&2
        sleep "$wait_seconds"
    done
}
