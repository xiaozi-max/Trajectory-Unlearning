#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$root/common_utils/experiment_env.sh"
trainer_env="${TRAINER_ENV:-$root/work/envs/trainer-py312}"
method="${1:?usage: run_mt_online.sh grpo|girpo|jev_girpo [hydra overrides...]}"
shift
case "$method" in
    grpo|girpo|jev_girpo) ;;
    *) echo "invalid method: $method" >&2; exit 2 ;;
esac

: "${TRAIN_MANIFEST:?set TRAIN_MANIFEST}"
: "${VAL_MANIFEST:?set VAL_MANIFEST}"
if [[ "$method" != grpo ]]; then
    : "${FORGET_POOL:?set FORGET_POOL for GiRPO methods}"
fi

driver_dir="${DRIVER_DIR:-$root/work/verl_driver}"
train_batch_size="${TRAIN_BATCH_SIZE:-8}"
val_batch_size="${VAL_BATCH_SIZE:-8}"
manifest_rows() { awk 'NF { n++ } END { print n + 0 }' "$1"; }
train_driver_size="${TRAIN_DRIVER_SIZE:-$(manifest_rows "$TRAIN_MANIFEST")}"
val_driver_size="${VAL_DRIVER_SIZE:-$(manifest_rows "$VAL_MANIFEST")}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
output_dir="${OUTPUT_DIR:-$root/work/runs/${method}/${run_id}}"
mkdir -p "$output_dir/rollouts" "$output_dir/validation" "$output_dir/checkpoints"
export PYTHONUNBUFFERED=1
echo "initializing" > "$output_dir/phase"
if [[ "${RISK_GIRPO_TIMESTAMPED:-0}" != 1 ]]; then
    export RISK_GIRPO_TIMESTAMPED=1 RUN_ID="$run_id"
    progress_heartbeat "$output_dir" &
    heartbeat_pid=$!
    trap 'kill "$heartbeat_pid" 2>/dev/null || true' EXIT INT TERM
    set +e
    bash "$0" "$method" "$@" 2>&1 | awk 'BEGIN { RS="[\r\n]+" } { print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }' | tee "$output_dir/train.log"
    status=${PIPESTATUS[0]}
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    trap - EXIT INT TERM
    set -e
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] run finished status=$status log=$output_dir/train.log" | tee -a "$output_dir/train.log"
    exit "$status"
fi
echo "planning" > "$output_dir/phase"
echo "[plan] method=$method run_id=$run_id GPUs=${GPU_COUNT:-4} train_batch=$train_batch_size group=${GROUP_SIZE:-4} max_steps=${MAX_STEPS:-30}"
echo "[plan] driver train_rows=$train_driver_size val_rows=$val_driver_size expected_optimizer_steps=$((train_driver_size / train_batch_size * ${EPOCHS:-1}))"
echo "[plan] upper-bound train environment actions=$((train_batch_size * ${GROUP_SIZE:-4} * ${MAX_STEPS:-30})); ETA becomes data-driven after first reset/step"
echo "[estimate] cold start=5-20m, train rollout=15-60m, optimization=2-10m, validation=5-20m; first pilot total roughly 30-90m"
echo "model-cache-check" > "$output_dir/phase"
model_path="${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}"
if [[ "$model_path" != /* ]]; then
    shopt -s nullglob
    model_snapshots=("$HF_HOME/hub/models--${model_path//\//--}/snapshots/"*)
    shopt -u nullglob
    if (( ${#model_snapshots[@]} > 0 )); then
        model_path="${model_snapshots[0]}"
        echo "[model] using cached snapshot: $model_path"
    else
        echo "[model] cache miss; Hugging Face download progress will follow for: $model_path"
    fi
fi
echo "building-driver" > "$output_dir/phase"
"$trainer_env/bin/python" "$root/scripts/build_verl_driver.py" \
    --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
    --output-dir "$driver_dir" --seed "${SEED:-20261101}" \
    --train-size "$train_driver_size" --val-size "$val_driver_size"

gpu_count="${GPU_COUNT:-4}"
if (( train_batch_size % gpu_count != 0 )); then
    echo "TRAIN_BATCH_SIZE ($train_batch_size) must be divisible by GPU_COUNT ($gpu_count)" >&2
    exit 2
fi
if (( train_driver_size < train_batch_size || train_driver_size % train_batch_size != 0 )); then
    echo "TRAIN_DRIVER_SIZE ($train_driver_size) must be >= and divisible by TRAIN_BATCH_SIZE ($train_batch_size)" >&2
    exit 2
fi
echo "waiting-for-gpus" > "$output_dir/phase"
grab_gpus "$gpu_count" "${GPU_MIN_MEMORY_MB:-30000}" "${GPU_MAX_UTIL:-10}" "${GPU_MAX_PROCESSES:-0}"

echo "$CUDA_VISIBLE_DEVICES" > "$output_dir/gpus"
echo "ray-training" > "$output_dir/phase"
echo "[launch] starting Ray/veRL trainer on GPU(s) $CUDA_VISIBLE_DEVICES"
exec "$trainer_env/bin/python" -u -m verl.trainer.main_ppo_mt_agentrisk \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.filter_groups.enable=false \
    algorithm.traj.enable_analysis=false \
    data.train_files="$driver_dir/train.parquet" \
    data.val_files="$driver_dir/validation.parquet" \
    data.train_batch_size="$train_batch_size" \
    data.val_batch_size="$val_batch_size" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=false \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.lora_rank="${LORA_RANK:-16}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-32}" \
    actor_rollout_ref.model.use_shm=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.01}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.rollout.multi_turn.enable=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    env.max_steps="${MAX_STEPS:-30}" \
    env.rollout.n="${GROUP_SIZE:-4}" \
    trainer.n_gpus_per_node="$gpu_count" \
    trainer.total_epochs="${EPOCHS:-1}" \
    trainer.val_before_train=false \
    trainer.logger="${TRAINER_LOGGER:-['console','wandb']}" \
    trainer.project_name="${WANDB_PROJECT:-risk-aware-girpo}" \
        trainer.experiment_name="${EXPERIMENT_NAME:-mt_${method}_${run_id}}" \
        trainer.default_local_dir="$output_dir/checkpoints" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-$output_dir/rollouts}" \
    trainer.validation_data_dir="${VALIDATION_DATA_DIR:-$output_dir/validation}" \
    trainer.log_val_generations="${LOG_VAL_GENERATIONS:-$val_batch_size}" \
    trainer.test_freq="${TEST_FREQ:-4}" \
    trainer.save_freq="${SAVE_FREQ:-4}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-1}" \
    +env.mt_agentrisk.gateway_url="${OPENHANDS_GATEWAY_URL:-http://127.0.0.1:8099}" \
    +env.mt_agentrisk.train_manifest="$TRAIN_MANIFEST" \
    +env.mt_agentrisk.val_manifest="$VAL_MANIFEST" \
    +unlearn.method="$method" \
    +unlearn.forget_trajectories_path="${FORGET_POOL:-}" \
    +unlearn.extra_penalty="${FORGET_MARGIN:-0.0}" \
    "$@"
