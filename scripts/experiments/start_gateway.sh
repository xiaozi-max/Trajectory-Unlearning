#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/common_utils/experiment_env.sh"
gateway_env="${GATEWAY_ENV:-$root/work/envs/gateway-py312}"

: "${TASKS_ROOT:?set TASKS_ROOT to the MT-AgentRisk task directory}"
: "${TASK_MANIFESTS:?set TASK_MANIFESTS to train/validation manifests separated by ':'}"
: "${AGENT_CONFIG_FILE:?set AGENT_CONFIG_FILE to a ToolShield TOML config}"

export PYTHONPATH="$root:$root/sources/ToolShield-main:$root/sources/OpenHands-0.54.0${PYTHONPATH:+:$PYTHONPATH}"
export TOOLSHIELD_REPO_ROOT="$root/sources/ToolShield-main"
export GATEWAY_WORK_ROOT="${GATEWAY_WORK_ROOT:-$root/work/gateway_sessions}"

exec "$gateway_env/bin/python" -m uvicorn risk_aware_girpo.gateway_server:create_app \
    --factory --host "${GATEWAY_HOST:-127.0.0.1}" --port "${GATEWAY_PORT:-8099}"
