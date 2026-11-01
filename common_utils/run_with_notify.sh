#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/experiment_env.sh"
name="$1"
shift
python "$RISK_GIRPO_ROOT/common_utils/notify.py" start "$name" || true
set +e
"$@"
status=$?
set -e
if [[ $status -eq 0 ]]; then
    python "$RISK_GIRPO_ROOT/common_utils/notify.py" success "$name" || true
else
    python "$RISK_GIRPO_ROOT/common_utils/notify.py" failure "$name" --detail "exit_code=$status" || true
fi
exit "$status"
