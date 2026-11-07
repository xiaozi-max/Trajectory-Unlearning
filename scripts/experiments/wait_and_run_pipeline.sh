#!/usr/bin/env bash
# Queue the full collect -> JEV -> risk-aware GiRPO pipeline until GPUs are free.
#
# Start (from the repo root):
#   nohup bash scripts/experiments/wait_and_run_pipeline.sh \
#     > work/logs/wait_and_run_pipeline.log 2>&1 & echo $! > work/logs/wait_pipeline.pid
# Inspect:
#   cat work/logs/wait_pipeline.phase ; tail -f work/logs/wait_and_run_pipeline.log
# Stop cleanly at any stage boundary:
#   touch work/STOP_PIPELINE
#
# GPU gating uses the same thresholds as common_utils/grab_gpus (called again
# inside run_mt_online.sh), so this loop only decides *when* to hand off; it
# never holds a GPU itself and never kills another user's process.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "$root/common_utils/experiment_env.sh"
mkdir -p "$root/work/logs" "$root/work/jev_cache" "$root/work/forget_pool"

phase_file="$root/work/logs/wait_pipeline.phase"
stop_file="$root/work/STOP_PIPELINE"
note() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$*" > "$phase_file"
}
abort_if_stopped() {
    if [[ -f "$stop_file" ]]; then
        note "stopped by $stop_file"
        exit 0
    fi
}

py="${TRAINER_ENV:-$root/work/envs/trainer-py312}/bin/python"
[[ -x "$py" ]] || py=python3
gpu_count="${GPU_COUNT:-2}"
gpu_ids="${GPU_IDS:-0 1 2 3 4 5 6 7}"
wait_s="${GPU_WAIT_SECONDS:-15}"

note "waiting for $gpu_count qualifying GPUs (free>=${GPU_MIN_MEMORY_MB:-30000}MB util<=${GPU_MAX_UTIL:-10}%)"
while true; do
    abort_if_stopped
    err="$(mktemp)"
    # shellcheck disable=SC2086
    selected="$("$py" "$root/common_utils/get_gpu_ids.py" -N "$gpu_count" \
        -M "${GPU_MIN_MEMORY_MB:-30000}" -U "${GPU_MAX_UTIL:-10}" \
        -P "${GPU_MAX_PROCESSES:-8}" -I $gpu_ids -v 2>"$err")" || true
    if [[ -n "$selected" ]]; then
        note "qualifying GPUs: $selected (launchers re-check with grab_gpus)"
        rm -f "$err"
        break
    fi
    note "no qualifying GPU [$(tr '\n' ' ' < "$err" | cut -c1-200)] retry in ${wait_s}s"
    rm -f "$err"
    sleep "$wait_s"
done

abort_if_stopped
note "gateway health check: ${OPENHANDS_GATEWAY_URL:-http://127.0.0.1:8099}"
curl -fsS --max-time 10 "${OPENHANDS_GATEWAY_URL:-http://127.0.0.1:8099}/openapi.json" >/dev/null \
    || { note "gateway unreachable; abort"; exit 1; }

note "stage 1/3 collect: frozen-base rollout + build forget pool"
bash "$root/scripts/experiments/collect_forget_pool.sh"

pool="$root/work/forget_pool/terminal_base_unsafe.jsonl"
[[ -s "$pool" ]] || { note "empty forget pool; abort before training"; exit 3; }
note "forget pool: $(wc -l < "$pool") dangerous attempt(s)"

note "stage 2/3 JEV: attempt judge, then per-step weights for dangerous attempts only"
scored="$root/work/forget_pool/terminal_base_unsafe_jev.jsonl"
"$py" "$root/scripts/score_jev.py" \
    --input "$pool" --output "$scored" \
    --cache-dir "$root/work/jev_cache" --model "${JEV_MODEL:-jev-latest}" \
    --attempts --steps

usable="$("$py" - "$scored" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.strip():
        continue
    j = json.loads(line).get("judge_output") or {}
    if j.get("outcome") == "COMPLETE" or float(j.get("unsafe_intermediate", 0)) > 0.5:
        n += 1
print(n)
PY
)"
[[ "$usable" -gt 0 ]] || { note "JEV confirmed 0 dangerous attempts; abort before training"; exit 4; }
note "JEV confirmed $usable dangerous attempt(s)"

note "stage 3/3 risk-aware GiRPO training"
export FORGET_POOL="$scored"
bash "$root/scripts/experiments/run_mt_online.sh" jev_girpo
note "pipeline finished"
