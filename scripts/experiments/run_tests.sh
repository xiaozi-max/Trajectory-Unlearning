#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/common_utils/experiment_env.sh"
trainer_env="${TRAINER_ENV:-$root/work/envs/trainer-py312}"
cd "$root"
exec "$trainer_env/bin/python" -m pytest -q tests/risk_aware_girpo "$@"
