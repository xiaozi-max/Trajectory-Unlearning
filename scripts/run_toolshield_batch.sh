#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/common_utils/experiment_env.sh"

python "$root/scripts/run_toolshield_batch.py" \
    --toolshield-root "${TOOLSHIELD_ROOT:?set TOOLSHIELD_ROOT}" \
    --tasks-root "${TASKS_ROOT:?set TASKS_ROOT}" \
    --manifest "${EVAL_MANIFEST:?set EVAL_MANIFEST}" \
    --outputs-root "${OUTPUTS_ROOT:?set OUTPUTS_ROOT}" \
    --agent-config-file "${AGENT_CONFIG_FILE:?set AGENT_CONFIG_FILE}" \
    --model-id "${MODEL_ID:?set MODEL_ID}" \
    --seeds ${EVAL_SEEDS:-0} \
    "$@"
