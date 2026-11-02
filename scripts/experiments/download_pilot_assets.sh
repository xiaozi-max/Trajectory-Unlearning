#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/common_utils/experiment_env.sh"
hf="$root/work/envs/trainer-py312/bin/hf"

: "${HF_TOKEN:?put your Hugging Face token in $root/.env after accepting the MT-AgentRisk terms}"
[[ -x "$hf" ]] || { echo "Run scripts/experiments/setup_envs.sh first." >&2; exit 2; }

dataset_dir="${MT_AGENTRISK_DIR:-$root/work/datasets/MT-AgentRisk}"
"$hf" download CHATS-Lab/MT-AgentRisk --repo-type dataset --local-dir "$dataset_dir"
echo "MT-AgentRisk downloaded to $dataset_dir"

if [[ "${DOWNLOAD_MODEL:-0}" == 1 ]]; then
    model_id="${MODEL_ID:-Qwen/Qwen2.5-Coder-3B-Instruct}"
    "$hf" download "$model_id"
    echo "$model_id downloaded to the HF cache at $HF_HOME"
fi
