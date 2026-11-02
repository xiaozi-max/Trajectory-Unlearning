#!/usr/bin/env bash
set -o pipefail

RISK_GIRPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RISK_GIRPO_ROOT

# Keep machine-local credentials and paths in one git-ignored file. Launchers
# source this helper, so users do not need to repeat exports in every shell.
if [[ -f "$RISK_GIRPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$RISK_GIRPO_ROOT/.env"
    set +a
fi

export HF_HOME="${HF_HOME:-$RISK_GIRPO_ROOT/work/huggingface}"
export WANDB_PROJECT="${WANDB_PROJECT:-risk-aware-girpo}"
export WANDB_DIR="${WANDB_DIR:-$RISK_GIRPO_ROOT/work/wandb}"
export TMPDIR="${TMPDIR:-$RISK_GIRPO_ROOT/work/tmp}"
mkdir -p "$HF_HOME" "$WANDB_DIR" "$TMPDIR"

grab_gpus() {
    local count="${1:-1}" memory="${2:-30000}" util="${3:-10}" max_proc="${4:-0}"
    local ids="${GPU_IDS:-0 1 2 3 4 5 6 7}" wait_seconds="${GPU_WAIT_SECONDS:-60}"
    local selected
    while true; do
        # shellcheck disable=SC2086
        selected="$(python "$RISK_GIRPO_ROOT/common_utils/get_gpu_ids.py" \
            -N "$count" -M "$memory" -U "$util" -P "$max_proc" -I $ids)"
        if [[ -n "$selected" ]]; then
            export CUDA_VISIBLE_DEVICES="$selected"
            echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
            return 0
        fi
        echo "No qualifying GPU; retrying in ${wait_seconds}s" >&2
        sleep "$wait_seconds"
    done
}
