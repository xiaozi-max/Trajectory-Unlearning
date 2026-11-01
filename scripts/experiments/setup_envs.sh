#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/common_utils/experiment_env.sh"
trainer_env="${TRAINER_ENV:-$root/work/envs/trainer-py312}"
gateway_env="${GATEWAY_ENV:-$root/work/envs/gateway-py312}"
# The host image pins unrelated serving packages through /etc/pip/constraint.txt.
# These project-local virtualenvs follow the upstream repos' own pins instead.
export PIP_CONSTRAINT=
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$root/work/pip-cache}"

[[ -d "$root/sources/OpenHands-0.54.0" ]] || {
    echo "Missing sources/OpenHands-0.54.0; clone the ToolShield-pinned tag first." >&2
    exit 2
}

python -m venv "$trainer_env"
"$trainer_env/bin/python" -m pip install --upgrade pip
"$trainer_env/bin/python" -m pip install -r "$root/requirements/mt-agentrisk-trainer.txt"
MAX_JOBS="${MAX_JOBS:-32}" FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80}" \
    "$trainer_env/bin/python" -m pip install flash-attn==2.7.4.post1 --no-build-isolation

cp "$root/sources/ToolShield-main/agentrisk/client.py" \
   "$root/sources/OpenHands-0.54.0/openhands/mcp/client.py"
python -m venv "$gateway_env"
"$gateway_env/bin/python" -m pip install --upgrade pip
"$gateway_env/bin/python" -m pip install -r "$root/requirements/mt-agentrisk-gateway.txt"

"$trainer_env/bin/python" "$root/scripts/check_trainer_env.py"
(cd "$root/work" && "$gateway_env/bin/python" -m pip check)
