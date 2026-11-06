#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/common_utils/experiment_env.sh"
trainer_env="${TRAINER_ENV:-$root/work/envs/trainer-py312}"
: "${DPO_ATTEMPTS:?set DPO_ATTEMPTS}"
: "${DPO_PAIRS:?set DPO_PAIRS}"
grab_gpus "${GPU_COUNT:-1}" "${GPU_MIN_MEMORY_MB:-30000}" "${GPU_MAX_UTIL:-10}" "${GPU_MAX_PROCESSES:-0}"
exec "$trainer_env/bin/python" -m risk_aware_girpo.train_trajectory_dpo \
    --attempts "$DPO_ATTEMPTS" --pairs "$DPO_PAIRS" \
    --model "${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}" \
    --output-dir "${OUTPUT_DIR:-$root/work/checkpoints/trajectory_dpo}" \
    --rank "${LORA_RANK:-16}" --learning-rate "${LEARNING_RATE:-1e-5}" \
    --epochs "${EPOCHS:-1}" "$@"
