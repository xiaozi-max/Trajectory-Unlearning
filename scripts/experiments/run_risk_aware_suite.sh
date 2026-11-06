#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$root/common_utils/experiment_env.sh"
domain="${1:-terminal}"; case "$domain" in terminal) ;; filesystem) echo "filesystem manifests are prepared, but parallel RL is blocked until per-session MCP isolation is configured" >&2; exit 2;; *) echo "usage: $0 terminal" >&2; exit 2;; esac
manifest_dir="${MANIFEST_DIR:-$root/work/manifests/research}"
export TRAIN_MANIFEST="$manifest_dir/${domain}_train.jsonl" VAL_MANIFEST="$manifest_dir/${domain}_validation.jsonl"
: "${FORGET_POOL:?Set FORGET_POOL to a frozen unsafe base-model pool}"
for method in grpo girpo jev_girpo; do echo "[suite] starting method=$method"; export EXPERIMENT_NAME="${domain}_${method}_${RUN_TAG:-$(date -u +%Y%m%d)}"; bash "$root/scripts/experiments/run_mt_online.sh" "$method"; done
