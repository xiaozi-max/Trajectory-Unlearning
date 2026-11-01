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
"$trainer_env/bin/python" "$root/scripts/build_verl_driver.py" \
    --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" \
    --output-dir "$driver_dir" --seed "${SEED:-20261101}"

gpu_count="${GPU_COUNT:-4}"
grab_gpus "$gpu_count" "${GPU_MIN_MEMORY_MB:-30000}" "${GPU_MAX_UTIL:-10}" "${GPU_MAX_PROCESSES:-0}"

exec "$trainer_env/bin/python" -m verl.trainer.main_ppo_mt_agentrisk \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.filter_groups.enable=false \
    data.train_files="$driver_dir/train.parquet" \
    data.val_files="$driver_dir/validation.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE:-8}" \
    data.val_batch_size="${VAL_BATCH_SIZE:-8}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-8192}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-1024}" \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=false \
    actor_rollout_ref.model.path="${MODEL_PATH:-Qwen/Qwen2.5-Coder-3B-Instruct}" \
    actor_rollout_ref.model.lora_rank="${LORA_RANK:-16}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-32}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE:-1e-6}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.01}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE:-1}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    env.max_steps="${MAX_STEPS:-30}" \
    env.rollout.n="${GROUP_SIZE:-4}" \
    trainer.n_gpus_per_node="$gpu_count" \
    trainer.total_epochs="${EPOCHS:-1}" \
    trainer.val_before_train=false \
    trainer.logger="${TRAINER_LOGGER:-['console','wandb']}" \
    trainer.project_name="${WANDB_PROJECT:-risk-aware-girpo}" \
    trainer.experiment_name="${EXPERIMENT_NAME:-mt_${method}}" \
    trainer.default_local_dir="${OUTPUT_DIR:-$root/work/checkpoints/$method}" \
    +env.mt_agentrisk.gateway_url="${OPENHANDS_GATEWAY_URL:-http://127.0.0.1:8099}" \
    +env.mt_agentrisk.train_manifest="$TRAIN_MANIFEST" \
    +env.mt_agentrisk.val_manifest="$VAL_MANIFEST" \
    +unlearn.method="$method" \
    +unlearn.forget_trajectories_path="${FORGET_POOL:-}" \
    +unlearn.extra_penalty="${FORGET_MARGIN:-0.0}" \
    "$@"
