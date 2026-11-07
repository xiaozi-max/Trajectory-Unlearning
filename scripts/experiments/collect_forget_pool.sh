#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$root/common_utils/experiment_env.sh"
manifest_dir="${MANIFEST_DIR:-$root/work/manifests/research}"
export TRAIN_MANIFEST="$manifest_dir/terminal_train.jsonl" VAL_MANIFEST="$manifest_dir/terminal_validation.jsonl"
export TRAIN_DRIVER_SIZE="${COLLECT_DRIVER_SIZE:-32}" EPOCHS=1 LEARNING_RATE=0 TEST_FREQ=-1 SAVE_FREQ=-1
export EXPERIMENT_NAME="terminal_forget_collection_${RUN_TAG:-$(date -u +%Y%m%d)}"
run_id="collect_$(date -u +%Y%m%dT%H%M%SZ)"; export RUN_ID="$run_id" OUTPUT_DIR="$root/work/runs/forget_collection/$run_id"
echo "[collect] starting frozen-base rollout collection: rows=$TRAIN_DRIVER_SIZE lr=$LEARNING_RATE"
bash "$root/scripts/experiments/run_mt_online.sh" grpo "$@"
echo "[collect] rollout finished; building forget pool"
mapfile -t rollouts < <(find "$OUTPUT_DIR/rollouts" -name '*.jsonl' -type f | sort)
"${TRAINER_ENV:-$root/work/envs/trainer-py312}/bin/python" "$root/scripts/build_forget_pool_from_rollouts.py" "${rollouts[@]}" --output "$root/work/forget_pool/terminal_base_unsafe.jsonl" --all-output "$root/work/train_attempts.jsonl"
echo "FORGET_POOL=$root/work/forget_pool/terminal_base_unsafe.jsonl"
