# Risk-aware GiRPO × MT-AgentRisk 代码报告

记录时间：2026-11-01 17:33 CST（Asia/Shanghai）

## 1. 选择的主仓库

最终代码位于：

```text
/volume/users/chenyun/lzc/risk-aware-girpo
```

选择 **Trajectory-Unlearning** 作为主仓库，而不是在 ToolShield 或 open-unlearning 上开发。原因是本项目最关键、风险最高的修改都在 GiRPO 的 pseudo rollout、real-only GRPO baseline 和 PPO actor loss；在原实现上做增量修改，可以继续复用其 tokenizer、response mask、LoRA、rollout/ref/actor 和 RayPPOTrainer 链路。

固定/参考版本：

| 代码 | 版本 |
| --- | --- |
| Trajectory-Unlearning 主干 | `2cde7a505b955f7bd7aece416b9118e68456af85` |
| ToolShield 源码快照 | `0.1.3` 的 main archive；不含 `.git`；`pyproject.toml` SHA-256 `01e2d429...` |
| open-unlearning 源码快照 | main archive；不含 `.git`；`README.md` SHA-256 `bf08f4b0...` |
| OpenHands | `0.54.0` / `4aae68ae53477d05010a509f1f1e3f02e61b1831`（ToolShield 官方固定版本） |

ToolShield/open-unlearning 是下载的源码 archive，没有自己的 Git metadata，因此不虚构 commit ID；
它们只用于接口核对并位于忽略目录 `sources/`。OpenHands 则是可验证的 detached tag clone。

### 1.1 原始 Trajectory-Unlearning / GiRPO 在做什么

原仓库把 veRL 的一次 PPO rollout 改成“模型—环境”多轮交互：
`TrajectoryCollector` 重复调用 actor 生成、环境 `step` 和 tokenizer 拼接，最终形成带
`response_mask` 的整条 trajectory。GiRPO 在每个真实 GRPO group 后注入一条已经成功执行过
目标行为的 forget trajectory；group 的均值/方差只由真实 trajectories 计算，注入行获得负向
advantage，然后与真实行一起进入同一个 PPO actor loss。这样模型在保持在线任务学习的同时，
对指定行为的 token 概率施加反向更新。

原实现主要面向 ALFWorld/WebShop，忘却轨迹靠 observation/game file 匹配，而且整条注入轨迹
权重相同。本项目保留“real-only statistics → injection → PPO loss”主干，只把 provenance 改成
MT-AgentRisk 的 `task_id`，并在 actor loss 前给注入行逐 decision step 乘 JEV 权重。真实行、
`response_mask`、loss aggregation 分母和 ref/KL 链路均不改变；因此全 1 权重严格退化为原 GiRPO。

## 2. 实现结构

```text
ToolShield/OpenHands tasks
        │ exact prompt/response/action/observation
        ▼
统一 attempt JSONL ── JEV step/attempt cache
        │                    │
        │                    └─ step_weight ∈ [0, 1]
        ▼
task_id → [verified COMPLETE trajectories]
        │ round-robin
        ▼
MTAgentRiskForgetCollector
        │ real rows + one injected trajectory/group
        ▼
real-only GRPO mean/std → injected negative advantage
        │
        ▼
actor loss 前 advantage × step_weight
        │ response_mask 和 loss 分母不变
        ▼
GRPO / GiRPO / JEV-GiRPO（同一训练链）
```

### 2.1 统一数据与 ToolShield 转换

- `risk_aware_girpo/schema.py`：定义 attempt/decision step schema；强制连续 step ID、合法 split 和 `[0,1]` 权重。
- `risk_aware_girpo/toolshield.py`：把结构化 ToolShield/OpenHands 轨迹转换为统一 JSONL。为了保证训练可重算 token IDs/log-probs，若轨迹缺少 exact prompt、model response 或 parsed action，会直接报错，不从字符串 state dump 猜测。
- `scripts/convert_toolshield.py`：单个 attempt 转换入口。
- `scripts/run_toolshield_batch.py` / `.sh`：对固定 manifest 和 seeds 独立运行 base/训练后模型评测。

统一行保留：`task_id, split, attempt_id, seed, model_id, adapter_id, turn_id, step_id, prompt, model_response, action, observation, terminal_state, evaluator_evidence, judge_output`。

### 2.2 JEV

- `risk_aware_girpo/jev.py`：实现研究计划中的两个固定英文请求模板。
- 请求只有 `model/state/questions`，没有额外 system prompt。
- cache key 为模板版本和 canonical JSON 的 SHA-256；缓存同时保存原请求、原响应、模型和模板版本。
- step Noul 严格检查 `[0,1]`；attempt Choice 严格限制为 `COMPLETE/REJECT/FAILED`。
- `scripts/score_jev.py` 支持 `--steps` 与 `--attempts`，后续 observation 不会进入当前 step 的评分请求。

正式实验时先用 `jev-latest` 连通，然后把 `--model` 固定为 `/v1/models` 返回的具体版本。

### 2.3 GiRPO / JEV-GiRPO

- `risk_aware_girpo/forget_pool.py`：只接受 train split；只保留 verified COMPLETE 或有核实 unsafe intermediate 的 attempts；同一 task 的多条轨迹轮换。
- `risk_aware_girpo/injector.py`：以 `task_id` 匹配，不再使用 ALFWorld 的 observation/game file；注入行携带 `trajectory_id, forget_step_id, forget_step_weight`。
- `agent_system/multi_turn_rollout/rollout_loop.py`：把环境 observation 中的 `task_id` 传入 batch。
- `unlearn/unlearn_inject.py`：声明 `task_id` 是 injector 可安全回填的 provenance 字段。
- `risk_aware_girpo/weighting.py`：在优势计算完成后，把每个 decision step 的标量权重扩展到有效 response token 宽度。
- `verl/trainer/ppo/ray_trainer.py`：real-only GRPO mean/std 完成后才附加 `policy_weights`。
- `verl/workers/actor/dp_actor.py` 与 `megatron_actor.py`：调用 PPO loss 前执行 `advantages *= policy_weights`。不改 response mask，不改 loss aggregation 分母，不改 real rows。

数值退化关系：

- GRPO：不启用 injector；
- GiRPO：启用 injector，所有 injected weights 为 1；
- JEV-GiRPO：同一 forget pool、同一 real rollout 预算，只读取缓存的 JEV weights；
- JEV-GiRPO 权重全 1 时与 GiRPO 数值一致；单步权重 0 时只有该 injected step 的直接忘却项为 0。

训练入口：`verl/trainer/main_ppo_mt_agentrisk.py`。

### 2.4 MT-AgentRisk / OpenHands 环境边界

`risk_aware_girpo/mt_environment.py` 定义 client，`risk_aware_girpo/gateway.py` 与
`gateway_server.py` 实现 stateful OpenHands gateway：

```text
POST /reset {task_id, seed}
  -> {session_id, observation, info}

POST /step {session_id, action}
  -> {observation, reward, done, won, terminal_state, evaluator_evidence, info}

POST /close {session_id}
```

veRL actor 负责每次模型生成；gateway 只拥有 OpenHands runtime、MCP 服务和 sandbox 状态。因此一次 actor update 后，下一轮 rollout 自动使用新 LoRA，不需要把模型生成再交给 OpenHands 内置 LLM。

gateway 只接受三个明确动作：`bash`、`mcp`、`finish/reject`，并转换为
OpenHands 0.54.0 的 serialized action。`risk_aware_girpo/openhands_runner.py` 延迟导入
ToolShield/OpenHands、按 task dependencies 生成 session-local MCP 配置，并把 workspace、
file store、运行记录都限制在 `work/gateway/`。session 目录会保留用于审计；实现不会自动清理。

`scripts/build_verl_driver.py` 把 train/validation manifest 转成 veRL 所需 parquet，写入
`task_id`、seed 与 `env_kwargs`，从而避免原 launcher 回落到 GSM8K 默认数据。真实 evaluator
仍由 ToolShield task 包提供；本仓库没有伪造 evaluator 结果。

### 2.5 SimNPO

- `risk_aware_girpo/train_simnpo.py`：Qwen2.5-Coder-3B-Instruct + LoRA 的独立离线 baseline。
- forget/retain 都来自统一 attempt JSONL。
- forget term 与 open-unlearning 一致：response token NLL 先按长度归一化，再计算 `-2/beta * log sigmoid(beta * (nll - delta))`。
- retain term 为标准 masked causal LM loss；总损失为 `gamma * forget + alpha * retain`。
- `scripts/run_simnpo.sh` 是实验入口。

### 2.6 Reward

`risk_aware_girpo/reward.py` 固化研究计划口径：

| 任务 | COMPLETE | REJECT | FAILED（无副作用） | 已有害副作用 |
| --- | ---: | ---: | ---: | ---: |
| harmful | -1 | +1 | 0 且 actor advantage 置零 | -1 |
| benign | +1 | -1 | 0 且 actor advantage 置零 | 按失败/任务证据复核 |

真实 gateway 应把此 reward decision 写入 step 终局 reward；JEV 与环境证据冲突的 attempt 不应自动进入训练池。

## 3. common_utils 适配

最终仓库内的 `common_utils/` 不包含任何 token 或 webhook：

- `experiment_env.sh`：把 HF cache、wandb、TMPDIR 全部限制在本项目 `work/`；提供 GPU 空闲轮询；
- `get_gpu_ids.py`：复用已有 GPU 管理逻辑；
- `notify.py`：从环境变量读取 `FEISHU_WEBHOOK_URL`，失败时由 wrapper fail-open；
- `run_with_notify.sh`：统一发送 start/success/failure。

实验脚本默认使用 `WANDB_PROJECT=risk-aware-girpo`。HF/W&B/飞书凭据只从已有环境读取，不写入仓库。

## 4. 如何运行

### 4.1 单元测试

```bash
cd /volume/users/chenyun/lzc/risk-aware-girpo
bash scripts/experiments/setup_envs.sh
bash scripts/experiments/run_tests.sh
```

环境目录是 `work/envs/trainer-py312` 与 `work/envs/gateway-py312`。pip cache、HF cache、
wandb、编译临时目录都在 `work/`。trainer 固定 `vllm==0.11.0`、
`transformers==4.57.3`、`pyarrow==19.0.1`，flash-attn 按 L40 的 8.x 架构编译；gateway
按 ToolShield 的要求安装 OpenHands 0.54.0。完整输入见 `requirements/`。

若数据包没有官方 split，先按 goal ID 分组、tool/attack type/benign 分层生成固定 manifest：

```bash
python scripts/build_manifests.py \
  --input /path/to/task_metadata.csv \
  --output-dir work/manifests \
  --seed 20261101
```

若输入已有合法 `train/validation/test` 字段，脚本优先保留官方 split。

### 4.2 ToolShield 独立 test rollout

```bash
export TOOLSHIELD_ROOT=/path/to/ToolShield
export TASKS_ROOT=/path/to/MT-AgentRisk/workspaces/terminal/multi_turn_tasks
export EVAL_MANIFEST=/path/to/test_manifest.jsonl
export OUTPUTS_ROOT=/volume/users/chenyun/lzc/risk-aware-girpo/work/eval
export AGENT_CONFIG_FILE=/path/to/agent_config.toml
export MODEL_ID=qwen25-base
export EVAL_SEEDS="101 102 103 104"
bash scripts/run_toolshield_batch.sh
```

每个模型必须独立运行该脚本；不能把 train rollout 当 base test。

### 4.3 JEV 缓存

```bash
python scripts/score_jev.py \
  --input work/train_attempts.jsonl \
  --output work/train_attempts_jev.jsonl \
  --cache-dir work/jev_cache \
  --model jev-latest \
  --steps --attempts
```

### 4.4 三个在线方法

先启动 gateway：

```bash
export TASKS_ROOT=/path/to/MT-AgentRisk/workspaces/terminal/multi_turn_tasks
export TASK_MANIFESTS=/path/to/task/manifests
export AGENT_CONFIG_FILE=/path/to/agent_config.toml
bash scripts/experiments/start_gateway.sh
```

再设置共同环境变量：

```bash
export MODEL_PATH=Qwen/Qwen2.5-Coder-3B-Instruct
export TRAIN_MANIFEST=/path/to/train_manifest.jsonl
export VAL_MANIFEST=/path/to/validation_manifest.jsonl
export FORGET_POOL=/path/to/train_attempts_jev.jsonl
export OPENHANDS_GATEWAY_URL=http://127.0.0.1:8099
export GPU_COUNT=4
```

运行：

```bash
bash scripts/experiments/run_mt_online.sh grpo
bash scripts/experiments/run_mt_online.sh girpo
bash scripts/experiments/run_mt_online.sh jev_girpo
```

推荐用通知 wrapper：

```bash
bash common_utils/run_with_notify.sh jev-girpo \
  bash scripts/experiments/run_mt_online.sh jev_girpo
```

### 4.5 SimNPO

```bash
export FORGET_POOL=/path/to/train_forget.jsonl
export RETAIN_POOL=/path/to/train_benign_success.jsonl
export OUTPUT_DIR=/volume/users/chenyun/lzc/risk-aware-girpo/work/checkpoints/simnpo
bash scripts/experiments/run_simnpo.sh
```

## 5. 测试记录

执行时间：2026-11-01 17:33 CST（远程服务器）。

```text
work/envs/trainer-py312/bin/python -m pytest -q tests/risk_aware_girpo
20 passed in 4.28s

scripts/experiments/setup_envs.sh（第二次幂等执行）
trainer environment: ok
gateway pip check: No broken requirements found

Hydra launcher --cfg job
passed（展开结果在 work/hydra-config-smoke.txt）

ToolShield/OpenHands import + serialized action adapter
passed

NVIDIA L40 flash-attn forward + backward（GPU 0，sm89）
passed，输出 shape=(2, 16, 4, 64)

bash -n + python -m py_compile
passed

git diff --check
passed
```

覆盖内容：

- JEV 请求模板、严格解析和 canonical hash；
- forget pool 过滤、多个 attempts round-robin；
- 全 1 权重恢复 GiRPO；
- 权重 0 只清除对应 injected step；
- 标量 step weight 正确扩展到 response tokens；
- MT 环境 reset/step/task_id 契约；
- harmful/benign reward 表；
- SimNPO 长度归一化。
- driver parquet schema 与 task_id/seed 传播；
- gateway 小动作协议、多轮推进、失败可重试、终局 reward；
- OpenHands runner 的 MCP config 生成和 task 文件复制边界；
- veRL `core_algos` 的真实 Ray 环境导入测试（无跳过）。

环境实际版本：Torch `2.8.0+cu128`、Ray `2.50.0`、vLLM `0.11.0`、
Transformers `4.57.3`、PEFT `0.21.2`、flash-attn `2.7.4.post1`、OpenHands `0.54.0`。

## 6. 当前外部阻塞与下一步验收

1. **MT-AgentRisk 未授权。** Hugging Face 返回 `403`，明确提示当前 token 不在 gated dataset authorized list。需要在数据集页面接受条款/申请访问，并把任务包放入本工作区后再运行。
2. **缺少真实 task package / evaluator。** gateway 与 OpenHands runner 已实现并完成 fake-runner 集成测试；没有 task package 时不能声称真实 MCP sandbox 已通过。
3. **真实训练仍需要数据后验收。** 安装前半段 8 张 L40 繁忙，因此没有抢占；17:33 时 0–3 号卡空闲，已在 GPU 0 完成 flash-attn 前后向 smoke test。获得 task package 后先跑 `2 tasks × 4 real attempts`，检查 adapter 更新、下一轮 rollout 权重变化和 reward 符号；再跑全 1 JEV 权重对 GiRPO 的数值对齐测试。

依赖环境、launcher、driver parquet、gateway server 都已在本工作区准备。以上外部条件满足后，才可以把状态标记为“真实 MT-AgentRisk 端到端训练通过”。当前不把 fake gateway、CPU 测试或静态导入冒充为真实 GPU/MCP 实验。

## 7. ponytail skill 说明

写代码前已从用户指定仓库安装并读取 ponytail skill，位置为
`/volume/users/chenyun/lzc/.codex/skills/ponytail`。它直接影响了实现取舍：复用 Trajectory-Unlearning
现有 injection/actor-loss 链路，只加 task provenance、权重与 gateway 边界；三种在线方法共用
一个 launcher；动作协议限制为当前任务所需的最小集合；每个新增分支都用小型单元/集成测试覆盖，
没有引入不必要的抽象层或防御性 fallback。

## 8. 删除与目录边界记录

- 所有新增源码、环境、缓存和测试产物均位于 `/volume/users/chenyun/lzc`。
- 没有执行 `rm`、`git clean`、环境清理或 session 清理。
- 一次初始 flash-attn 探测继承了系统 `/tmp`；发现后立即中止，之后统一通过
  `common_utils/experiment_env.sh` 将 `TMPDIR` 固定到本仓库 `work/tmp`。未对 `/tmp` 做任何后续读取、修改或删除。
- 两个被中止的旧编译环境/临时树不做人工删除；如需回收空间，必须先征得用户同意。
