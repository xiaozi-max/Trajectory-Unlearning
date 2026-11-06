# Risk-aware GiRPO × MT-AgentRisk 配置与运行记录

记录时间：2026-11-06。本文是 `1104配置进展.md` 之后的续篇，包含可复现的完整配置步骤、本轮会话处理过的故障，以及当前状态。

## 1. 架构与路径

```text
GPU 服务器  yjk-8gpu-0   8 × L40 46GB
/volume/users/chenyun/lzc/risk-aware-girpo
├── work/envs/trainer-py312       veRL / vLLM / Ray / LoRA / JEV
├── work/huggingface              Qwen2.5-Coder-3B-Instruct 缓存
├── work/manifests/research       terminal_{train,validation}.jsonl
└── work/runs/…                  rollouts / checkpoints / 日志
          │  HTTP 127.0.0.1:18099
          │  SSH 本地端口转发
          ▼
阿里云 ECS  139.196.231.63   4 核 / 14Gi 可见内存
/root/risk-aware-girpo
├── work/envs/gateway-py312
├── work/datasets/MT-AgentRisk/workspaces
└── FastAPI Gateway 127.0.0.1:8099 → OpenHands 0.54.0 → Docker sandbox
```

GPU 侧负责模型生成、训练和评分；ECS 侧只负责执行 `bash`/`mcp` 动作并返回 observation。
ECS 的 Gateway **只监听 127.0.0.1:8099**，公网 8099 不可达，因此 SSH 隧道是必需的。

## 2. SSH 密钥与隧道

原 `/root/.ssh/girpo_ecs` 私钥曾丢失（`/root` 属于容器层，会被清理）。新密钥对放在持久卷：

```text
/volume/users/chenyun/lzc/.ssh/girpo_ecs      权限 600
/volume/users/chenyun/lzc/.ssh/girpo_ecs.pub  权限 644
~/.ssh/girpo_ecs -> 上面私钥的软链接
```

当前公钥（已加入 ECS `/root/.ssh/authorized_keys`）：

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGMCelE6yNFpRG77zx62Jb7mdhzBhuWQxvpM5tEgK7Mw root@yjk-8gpu-0
```

注意：只有公钥无法认证，必须持有对应私钥。若换新机器，直接复用同一把私钥则 ECS 无需改动；若重新 `ssh-keygen`，必须把新公钥追加到 ECS 的 `authorized_keys`。

建立隧道：

```bash
ssh -NT -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
    -L 18099:127.0.0.1:8099 -i ~/.ssh/girpo_ecs root@139.196.231.63
```

长时间训练建议用「ssh 退出即重连」的守护：

```bash
setsid nohup bash -c 'while true; do
  ssh -NT -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
      -o ExitOnForwardFailure=yes -L 18099:127.0.0.1:8099 \
      -i ~/.ssh/girpo_ecs root@139.196.231.63
  sleep 5
done' > work/logs/ssh_tunnel.log 2>&1 < /dev/null &
```

验证（容器内 `ss -ltn` 看不到 18099 监听，必须用 curl 判断）：

```bash
curl -sS -m 8 -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18099/openapi.json   # 期望 200
ssh -i ~/.ssh/girpo_ecs root@139.196.231.63 'ss -ltn | grep 8099'
```

## 3. ECS Gateway 启动

```bash
cd /root/risk-aware-girpo
set -a; . ./.env; set +a
setsid nohup bash scripts/experiments/start_gateway.sh > work/gateway-final.log 2>&1 < /dev/null &
```

`.env` 中的必需变量：

```text
TASKS_ROOT=/root/risk-aware-girpo/work/datasets/MT-AgentRisk/workspaces
TASK_MANIFESTS=/root/risk-aware-girpo/work/manifests/research/terminal_train.jsonl:/root/risk-aware-girpo/work/manifests/research/terminal_validation.jsonl
AGENT_CONFIG_FILE=/root/risk-aware-girpo/configs/agent_config.toml
GATEWAY_HOST=127.0.0.1
GATEWAY_PORT=8099
```

若报 `[Errno 98] address already in use`，说明旧 uvicorn 还在服务，先 `kill` 旧进程再启动。
`/close` 会调用 `runtime.close()` 删除容器（已验证：冒烟后 `docker ps -q | wc -l` 回到 0）。

端到端冒烟（GPU 侧，经隧道）：

```bash
POST /reset {"task_id":"multi-turn_disk-space","seed":42}
POST /step  {"session_id":..., "action":"<action>{\"type\":\"bash\",\"command\":\"pwd\"}</action>"}
POST /close {"session_id":...}
```

`task_id` 必须用 `work/manifests/research/terminal_train.jsonl` 里的下划线命名（如 `multi-turn_disk-space`）。旧的 `work/test_gateway.py` 读 pilot manifest 的 `multi-turn-disk-space`（连字符），会返回 HTTP 400。

## 4. Python 环境（已验证完好）

`work/envs/trainer-py312` 为 Python 3.12.3：

```text
torch 2.8.0+cu128   vllm 0.11.0   ray 2.50.0   transformers 4.57.3
peft 0.21.2   flash_attn 2.7.4.post1   pyarrow 19.0.1   numpy 2.2.6
hydra 1.3.8   omegaconf 2.3.1   wandb 0.31.0
cuda_available True   device_count 8
```

宿主机 `uptime` 479 天，**没有重启过**；`/root` 下的 tmux、SSH key 消失是执行容器/namespace 切换所致，持久卷上的环境与代码不受影响。tmux 已重装为 3.4（`apt-get update && apt-get install -y tmux`）。

## 5. 关键 `.env` 配置

```text
OPENHANDS_GATEWAY_URL=http://127.0.0.1:18099
MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct
GPU_COUNT=2
TRAIN_BATCH_SIZE=4
TRAIN_DRIVER_SIZE=16
GROUP_SIZE=4        # 实际启动 override 为 2
MAX_STEPS=10        # 实际启动 override 为 6
TRAINER_LOGGER=["console","wandb"]
```

`.env` 含 HF / W&B / JEV / OpenRouter / 飞书密钥，已被 `.gitignore` 忽略，绝不入库。

## 6. 实验启动命令

按 `scripts` 要求先收集 rollout，再跑 JEV-GiRPO。为压缩时间、保护 4 核 16GB 的 ECS，本轮使用缩小规模：

```bash
cd /volume/users/chenyun/lzc/risk-aware-girpo
export COLLECT_DRIVER_SIZE=8 GPU_WAIT_SECONDS=15 OPENHANDS_GATEWAY_TIMEOUT=300
bash scripts/experiments/collect_forget_pool.sh \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=9216 \
  env.rollout.n=2 env.max_steps=6
```

`collect_forget_pool.sh` 完成后生成 `work/forget_pool/terminal_base_unsafe.jsonl`，随后：

```bash
export FORGET_POOL=$PWD/work/forget_pool/terminal_base_unsafe.jsonl
export TRAIN_MANIFEST=$PWD/work/manifests/research/terminal_train.jsonl
export VAL_MANIFEST=$PWD/work/manifests/research/terminal_validation.jsonl
bash scripts/experiments/run_mt_online.sh jev_girpo \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=9216 \
  env.rollout.n=2 env.max_steps=6
```

`scripts/experiments/run_risk_aware_suite.sh terminal` 会依次跑 grpo / girpo / jev_girpo，需要预先设置 `FORGET_POOL`。

## 7. 本轮会话处理过的故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `Permission denied (publickey)` | 旧私钥随 `/root` 被清 | 在持久卷生成新密钥对并软链到 `~/.ssh`；公钥加入 ECS `authorized_keys` |
| `start_gateway.sh` 报 `Errno 98` | 旧 uvicorn 仍在监听 8099 | 确认旧进程可用；重启时先 kill |
| `/reset` HTTP 400 `'multi-turn-disk-space'` | pilot manifest 与 ECS catalog 的 task_id 命名不一致 | 改用 research manifest 的 `multi-turn_disk-space` |
| `ValueError: Free memory 11.29/44.31 GiB < desired 0.5 (22.15 GiB)` | FSDP + vLLM 同卡争显存 | 降低 `gpu_memory_utilization`、启用 offload、限制 `max_model_len` |
| `ValueError: Free memory 6.02/44.31 GiB < desired 0.23 (10.19 GiB)` | 同上，0.23 仍不够 | 启用 `param_offload/optimizer_offload` 后不再出现 |
| `TimeoutError: timed out` | ECS OOM，Gateway 无响应（客户端 `OPENHANDS_GATEWAY_TIMEOUT` 默认 900s） | 重启 ECS、清理容器、重启 Gateway；override 为 300s 快速失败 |
| ECS SSH `Connection timed out during banner exchange` | 19 个残留 `openhands-runtime-*` 容器吃满 14Gi，内核无法为 sshd fork | 阿里云 Workbench 重启实例 + 清理孤儿容器 + `systemctl restart sshd` |
| `train.log` 长时间为空 | `tee` 块缓冲，不是训练卡死 | 看 Ray logs 或等日志刷出；`phase` 文件更实时 |
| tmux / SSH key 「消失」 | 执行容器换 namespace，非宿主机重启 | 持久卷存放密钥；必要时重装 tmux |

offload 是 **GPU 服务器**上 veRL 把参数搬到本机 CPU 内存，与 ECS 内存无关；ECS OOM 的根因是历史中断实验残留的 Docker 容器（`/close` 未被调用）。

Linux OOM killer 只杀单个进程，不会清空所有容器；内存耗尽时连 sshd fork 都做不到，实例进入「内核活着、用户态卡死」状态，不会自动恢复，必须人工清理或重启。

## 8. 监控

```bash
tmux attach -t girpo                      # 实时面板（若 session 随容器消失需重建）
tmux capture-pane -p -t girpo
tail -f work/logs/collect_forget_pool.launch4.log
tail -f work/logs/jev_girpo.launch.log
tail -f work/runs/forget_collection/<run_id>/train.log
cat work/runs/forget_collection/<run_id>/phase
```

面板脚本在 `work/logs/dashboard.sh`，自愈隧道在 `work/logs/ssh_tunnel.log` 对应的循环中。

## 9. 已知限制与后续

- `scripts/build_forget_pool_from_rollouts.py` 把 `jev_weight` 写死为 `1.0`，且只保留 `episode_rewards < 0` 的轨迹。直接跑 `jev_girpo` 会**数值退化成普通 GiRPO**；要体现方法本身，需在两步之间执行 `python scripts/score_jev.py --input … --output … --cache-dir work/jev_cache --model jev-latest --steps --attempts`。
- `env.max_steps=6` 若让 harmful 任务全部 FAILED（reward 0），forget pool 会为空并报错，需放大步数重跑。
- 8 张 GPU 常被同机的 `yjk` vLLM 服务占满，`grab_gpus` 会轮询等待 2 张空卡（要求每卡空闲 ≥ 30GB、util ≤ 10%）。若换到 80GB 卡，可去掉 offload 并把 `gpu_memory_utilization` 调回 0.5–0.6。
- ECS 4 核 16GB 下并发 sandbox 保持 ≤ 2。

## 10. 代码改动

- `scripts/experiments/collect_forget_pool.sh` 第 11 行改为 `bash … run_mt_online.sh grpo "$@"`，使 Hydra override 可透传（本轮唯一修改的仓库脚本）。
- 其余为 `work/logs/` 下的临时脚本：`chain_jev_girpo.sh`（自动衔接）、`dashboard.sh`（监控面板）、`gateway_smoke2.py`（真实 task_id 冒烟）。

## 11. 迁移到另一台 GPU 服务器

仓库已忽略 `.env`、`work/`、`sources/`、`configs/mt_agentrisk/agent_config.toml`、`wandb/`、`outputs/`，密钥与大数据不会入库。

```bash
# 推送（origin 已指向自己的 fork）
git add -A && git commit -m "..." && git push origin main
```

新机器 clone 后需补齐：

1. `bash scripts/experiments/setup_envs.sh` 重建 `work/envs/trainer-py312`，用第 4 节的 import 检查验证。
2. 模型权重下载到 `work/huggingface`，或设置 `MODEL_PATH`。
3. 单独 scp（不进 git）：`.env`、`work/manifests/`、`configs/mt_agentrisk/agent_config.toml`（仅跑 Gateway 时需要）、`sources/`（仅跑 Gateway 时需要）。
4. 复用 `.ssh/girpo_ecs` 私钥（或把新公钥加到 ECS `authorized_keys`），按第 2 节建隧道，按第 3 节验证 Gateway。
5. 按第 6 节启动。多台 GPU 服务器可各建各的 18099 隧道连同一 ECS，但共享 Gateway，注意并发 sandbox 上限。
