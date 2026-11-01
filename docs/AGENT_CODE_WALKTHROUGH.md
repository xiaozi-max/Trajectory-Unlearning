# 从一次 Agent rollout 读懂 Risk-aware GiRPO

本仓库训练的是一个会反复观察、调用工具、读取工具结果并继续行动的策略。理解代码的最好方法，是跟随一条任务记录从输入文件走到一次参数更新。

```text
task manifest
  → build_verl_driver.py
  → main_ppo_mt_agentrisk.py
  → TrajectoryCollector
  → MTAgentRiskEnvironmentManager ──HTTP──> GatewayService / OpenHands
  → real trajectories and terminal rewards
  → GRPO advantages
  → optional GiRPO synthetic trajectory and optional JEV weights
  → PPO actor loss → updated model / LoRA
```

下面先说明这条链，再把文件放回链上的位置。

## 一、从任务文件到训练 batch

训练脚本是 `scripts/experiments/run_mt_online.sh`。它接收 `grpo`、`girpo` 或 `jev_girpo`，先运行 `scripts/build_verl_driver.py`。后者将 train/validation manifest 写成 veRL 所需的 parquet；一行数据至少携带 `task_id`、seed 和 `env_kwargs`。

`task_id` 是整套实现的主键。它在三个地方必须保持一致：manifest 指向要运行的任务目录；在线环境把它放进每个 trajectory row；forget pool 用它找到同题的历史危险轨迹。原仓库的 ALFWorld/WebShop 实验常以初始 observation 或 game file 连接在线与离线数据；本项目改为 `task_id`，避免多轮任务中由动态文本造成的不稳定匹配。

脚本随后启动 `verl/trainer/main_ppo_mt_agentrisk.py`。这个文件是装配入口：它创建 Ray worker、tokenizer、PPO trainer、环境和 trajectory collector。`unlearn.method=grpo` 时传入原始 collector；`girpo` 与 `jev_girpo` 时，先读 `FORGET_POOL`，再用 `MTAgentRiskForgetCollector` 包住原始 collector。三种模式的在线环境和真实样本预算相同，区别在于是否插入历史轨迹，以及插入轨迹是否使用 JEV 权重。

`env.rollout.n` 才是 GRPO group 的大小：每个任务要采集多少条真实 episode。`actor_rollout_ref.rollout.n` 固定为 1，因为它是 veRL 的普通重复生成参数；入口中的断言刻意把这两个概念分开。

## 二、一次 Agent episode 实际如何运行

`agent_system/multi_turn_rollout/rollout_loop.py` 中的 `TrajectoryCollector` 驱动多轮循环。它对仍未结束的 episode 执行四件事：根据当前 observation 建 prompt；要求 actor 生成 response；把 response 交给环境；将新 observation、reward、done 和元数据追加到 trajectory。结束后它把多轮数据拼成 veRL 的 `DataProto` batch。

一条 batch row 不只是字符串。`input_ids`、`attention_mask`、`position_ids` 和 `responses` 为模型前向和 log probability 服务；`response_mask` 指出 response 中哪些 token 是策略生成的。PPO 的 policy loss 只在 mask 为真的 token 上计算。GiRPO 注入历史轨迹时必须重建这些字段，所以只保存工具状态或最终摘要不够。

环境的入口是 `risk_aware_girpo/mt_environment.py`。`MTAgentRiskVectorEnv.reset` 调用 gateway 的 `/reset`，保存每个返回的 `session_id`，并向 collector 返回初始 observation 与 `task_id`。`step` 将一批模型 response 发到 `/step`，接收新 observation、reward、done、终局状态和证据。`MTAgentRiskEnvironmentManager` 负责把这些结果包装回现有 `agent_system` 期望的 `text`、`anchor`、`task_id` 字段。它使训练主循环不需要知道 HTTP、MCP 或 OpenHands 的接口细节。

gateway 位于 `risk_aware_girpo/gateway.py`。模型必须生成一个 `<action>{...}</action>` 块。`parse_actor_action` 将 `bash` 转成命令执行，将 `mcp` 转成 MCP 调用，并处理 `finish` 与 `reject`。格式错误不会终止 episode：gateway 返回带正确格式说明的 observation，让下一次模型生成修正动作。

`GatewayService.reset` 通过 `TaskCatalog` 找到任务目录，创建一个 session 目录及 runner。`TaskPlan.from_task_dir` 读取 `turns.yml`、`task-intro.md` 和各 turn 指令；`transition` 根据工具 observation 中的成功模式推进到下一 turn，或给出 COMPLETE、REJECT、FAILED。`risk_aware_girpo/openhands_runner.py` 的 `ToolShieldRunner` 是真正连接 OpenHands 的层：它为 session 写 MCP 配置、复制任务文件，并将规范化 action 交给运行时。换言之，veRL actor 决定动作内容，OpenHands 只负责在任务 sandbox 中执行；actor 更新后，下一次 rollout 自动使用新权重。

## 三、真实轨迹如何产生 GRPO 更新

终局 outcome 进入 `risk_aware_girpo/reward.py::outcome_reward`。危险任务的 `REJECT` 是 +1、`COMPLETE` 是 -1；正常任务方向相反；无副作用的 `FAILED` 为 0。环境将 reward 交给 `EpisodeRewardManager`，再交由 `RayPPOTrainer`。

GRPO 的单位是同题的一组真实 trajectory。`verl/trainer/ppo/ray_trainer.py` 使用这一组回报计算均值、方差和相对 advantage：高于组内平均的 response token 被提高概率，低于平均的 token 被降低概率。随后 `verl/workers/actor/dp_actor.py`（FSDP）或 `megatron_actor.py`（Megatron）用 PPO clipped loss、以及配置启用时的 reference-policy KL，更新 actor。veRL 的原始实现承担这部分优化工作。

## 四、GiRPO 如何让历史危险行为进入同一轮更新

离线轨迹先由 `risk_aware_girpo/toolshield.py` 转成统一 attempt JSONL。转换器要求保存每一步的 exact prompt、model response 和 action。`schema.py` 定义 attempt/step 结构和 JSONL 读写；`splits.py` 保证 train/validation/test 按任务分组，避免同一任务的不同 attempt 跨 split 泄漏。

`forget_pool.py::load_forget_pool` 只接受 `train` 行，保留判为 `COMPLETE` 或存在已核实危险中间效果的记录，并按 `task_id` 分组。`RoundRobinForgetPool` 在同一任务的多条记录间轮换。它的作用很小但很关键：collector 只负责“本轮为这个 task 选哪条”，不需要理解 JSONL、split 或筛选规则。

`injector.py::MTAgentRiskForgetCollector` 是本项目对原 GiRPO 的核心改写。它继承 `unlearn/unlearn_inject.py::ForgetOnlyBaselineExcludedAdvantageCollector`，复用父类中最容易出错的 token 化、response mask 和 synthetic row 构造逻辑。子类只改动 MT-AgentRisk 特有的部分：

1. `_select_for_batch` 从真实 row 读取 `task_id`，并在 episode 的 `step_num == 0` 时取一条历史轨迹。
2. `_inject` 调用父类，将历史 prompt/response 变成与真实 row 结构相同的 synthetic row。
3. 它为注入 row 写入 `trajectory_id`、`forget_step_id`、`forget_step_weight`，并标记 `data_source=mt_agentrisk`。

父类以 `is_forget_injected` 区分 synthetic 和真实 row。真实 rows 的 reward、mask 与 metadata 不被修改。训练器先用真实 rows 计算 GRPO 统计量，再给注入 row 附加遗忘所需的负向目标；这就是“real-only statistics”的含义。若把 synthetic row 也用于均值和方差，它会反过来改变真实样本的 baseline，算法就不再是同一个对照。

## 五、JEV 权重具体改了哪一行 loss

`risk_aware_girpo/jev.py` 先离线评分。`step_request` 只发送当前决策之前的历史、当前 response 和 action，得到 `suppress_this_step.noul`；`attempt_request` 发送完整轨迹和终局证据，得到 outcome 与 `unsafe_intermediate`。`JEVClient` 用“模板版本 + canonical JSON”的 hash 缓存响应，因此重跑不会因为同一请求被重新评分而悄悄改变训练输入。

在 `jev_girpo` 中，injector 将每个注入 step 的 `jev_weight` 写到 `forget_step_weight`。`weighting.py::attach_policy_weights` 将这个每行标量扩成与 `advantages` 同形状的 token 矩阵。最终 actor 的两条实现都在 PPO policy loss 前执行：

```python
advantages = advantages * policy_weights
```

它只缩放注入步骤的 policy-gradient advantage；不改 response mask，不改 token 分母。权重为 1 时等价于普通 GiRPO；权重为 0 时该步的直接 policy advantage 消失。若启用了 KL loss，KL 项仍按原配置计算。

## 六、本项目增加了什么，哪些代码属于原主线

| 代码 | 在 pipeline 中的作用 |
| --- | --- |
| `risk_aware_girpo/mt_environment.py`、`gateway.py`、`openhands_runner.py` | 将 MT-AgentRisk/ OpenHands 接成 veRL 多轮环境 |
| `schema.py`、`toolshield.py`、`forget_pool.py` | 制作可重放的轨迹数据，并用 task ID 供注入查询 |
| `jev.py`、`weighting.py` | 风险评分缓存与逐 token 权重 |
| `injector.py` | 在旧 GiRPO token 构造之上实现 task-ID 注入 |
| `main_ppo_mt_agentrisk.py` | 将环境、forget pool 和 veRL trainer 组装成新入口 |
| `ray_trainer.py`、`dp_actor.py`、`megatron_actor.py` 的改动 | 将 policy weights 从 advantage 计算一路送到两种 actor loss |

原主线仍承担大部分困难工作：`TrajectoryCollector` 维护多轮状态；`unlearn/unlearn_inject.py` 构造能参与 PPO 的 synthetic trajectory；veRL 计算 log probability、GRPO advantage、PPO 和 KL。Risk-aware GiRPO 没有复制这些实现，而是在它们的输入与连接点上加入 MT-AgentRisk task provenance、历史风险轨迹和 JEV 权重。

`simnpo.py` 与 `train_simnpo.py` 是另一条离线 baseline：它读取 forget/retain 对，计算 response NLL 的 SimNPO 目标并训练 LoRA，不使用 gateway、GRPO group 或在线 rollout。阅读和比较实验时应把它和三种在线方法分开。

## 七、目前需要特别核验的两处

gateway 当前通过 `turns.yml` 的 `success_patterns` 判断任务是否完成，`evaluator_evidence` 也由该 transition 形成。它足以驱动训练调试；正式评测仍应使用 `scripts/run_toolshield_batch.sh` 运行独立评测并保存外部 evaluator 证据。

此外，`RewardDecision.zero_actor_advantage` 会标记无副作用的 `FAILED`，但 `GatewayService.step` 目前只输出 `decision.reward`，没有把这个标记传到 actor advantage。因此它当前是尚未接线的设计意图，不是已生效的训练行为。

---

## 文件索引

下面提供按文件查找信息的索引。需要实验设置、运行命令和版本记录时，再阅读 [实现报告](MT_AGENTRISK_IMPLEMENTATION_REPORT.md)。

## 1. 先建立整体模型

这里的 agent 是一个反复执行“观察环境 → LLM 生成动作 → 工具执行 → 收到新观察”的策略。它不是把任务一次性发给 OpenHands 的内置模型：在线训练时，**veRL actor 每一步生成文本**，gateway 负责把文本里的动作交给 OpenHands 工具运行时执行。一个任务的多步交互构成 trajectory，多个同题 trajectory 构成 GRPO group。

```text
任务 manifest → veRL 数据集 → TrajectoryCollector → actor(vLLM/LoRA)
                                      ↕
                         MTAgentRiskEnvironmentManager
                                      ↕ HTTP
                     gateway → OpenHands runtime / MCP / bash
                                      ↓
                          真实 trajectory 与环境 reward
                                      ↓
              [GiRPO 可选] 按 task_id 注入旧的危险 trajectory
                                      ↓
                  real-only GRPO advantage → PPO actor loss
                                      ↑
                 [JEV 可选] 注入步骤的 token 权重
```

这条主线由 `scripts/experiments/run_mt_online.sh` 启动，进入 `verl/trainer/main_ppo_mt_agentrisk.py`。`risk_aware_girpo/` 是本项目新增的接合层；`agent_system/` 和 `verl/` 是原有 Agent 轨迹与 RL 训练骨架；`unlearn/` 保留原 GiRPO 注入实现及其他历史方法。

## 2. 经典框架和算法各自负责什么

| 名称 | 本仓库用法 | 先读哪里 |
| --- | --- | --- |
| **veRL** | Ray 上的 actor、rollout、reference policy、PPO 优化与数据协议；使用 Hydra 配置 | `verl/trainer/main_ppo_mt_agentrisk.py`、`verl/trainer/ppo/ray_trainer.py` |
| **GRPO** | 同一任务采样多个真实 trajectory，用组内奖励估计相对优势；`env.rollout.n` 是这里的组大小 | `agent_system/multi_turn_rollout/rollout_loop.py`、`ray_trainer.py` |
| **PPO** | actor 对已有 rollout 计算 clipped policy loss，可叠加 reference KL；GRPO 在这里提供 advantage，并不另造一个优化器 | `verl/workers/actor/dp_actor.py`、`megatron_actor.py` |
| **GiRPO** | 给匹配任务额外注入一条已记录的危险轨迹，以负向 advantage 降低其动作概率；组统计只用真实样本 | `unlearn/unlearn_inject.py`、`risk_aware_girpo/injector.py` |
| **OpenHands** | 工具和沙箱运行时，执行 bash/MCP 动作；在线模型生成仍由 veRL 负责 | `risk_aware_girpo/openhands_runner.py`、`gateway.py` |
| **ToolShield / MT-AgentRisk** | 多轮风险任务、轨迹及评测资料的来源；本仓库只实现部分格式转换与在线适配 | `risk_aware_girpo/toolshield.py`、`scripts/convert_toolshield.py` |
| **JEV** | 离线给危险轨迹的每个决策步骤打分，再用分数调节注入项强度 | `risk_aware_girpo/jev.py`、`scripts/score_jev.py` |
| **SimNPO** | 独立的离线遗忘基线，使用 retain/forget 文本对；不经过在线 gateway/GRPO | `risk_aware_girpo/simnpo.py`、`train_simnpo.py` |

Agent loop 由 `TrajectoryCollector` 与环境管理器组成；Ray/veRL 负责分布式 RL，OpenHands 负责工具执行。

## 3. 从任务到一次真实 rollout

### 3.1 任务和 veRL 输入

`scripts/build_verl_driver.py` 将 train/validation manifest 转成 veRL 能读取的 parquet；它保留 `task_id`、seed、`env_kwargs`。这一步很关键：训练样本必须能和 gateway 任务目录、forget pool 中的 `task_id` 对上。`scripts/build_manifests.py` 负责创建或保留固定 split；数据泄漏检查应以任务 ID 为单位，而非单条 attempt 为单位。

`run_mt_online.sh` 设置 `algorithm.adv_estimator=grpo`、`actor_rollout_ref.rollout.n=1`、`env.rollout.n=GROUP_SIZE`。前者是 veRL 普通生成重复数，后者才是环境里同题的真实采样数。脚本通过 `+unlearn.method` 在 `grpo`、`girpo`、`jev_girpo` 间切换。

### 3.2 环境 reset、step 与动作格式

`main_ppo_mt_agentrisk.py` 创建 `TrajectoryCollector` 和 `make_mt_agentrisk_envs(config)`，传给 `RayPPOTrainer`。`risk_aware_girpo/mt_environment.py` 把 veRL 需要的批量 reset/step 转为 gateway HTTP `/reset`、`/step`、`/close`。reset 返回初始 observation 和 `task_id`；step 返回新 observation、reward、done、won 以及终局证据。`agent_system/multi_turn_rollout/rollout_loop.py` 把 `task_id` 穿过每个 rollout row，供后续注入匹配。

模型的 response 必须含一个 `<action>{JSON}</action>` 块。例如：

```text
<action>{"type":"bash","command":"pwd"}</action>
```

`gateway.parse_actor_action` 支持 `bash`、`mcp`、`finish`、`reject`；`GatewayService.step` 执行动作并交给 `TaskPlan.transition` 推进 turn。`ToolShieldRunner` 再把动作转成 OpenHands action。一次 episode 结束后，`TrajectoryCollector` 将各步 prompt/response、mask、reward 拼成 veRL 的 `DataProto`。其中 `response_mask` 指出真正由策略生成、参与 loss 的 token。

### 3.3 reward 的含义

`risk_aware_girpo/reward.py` 定义 harmful/benign × COMPLETE/REJECT/FAILED 的终局 reward：危险任务拒绝为正、完成为负；正常任务完成为正、拒绝为负。`EpisodeRewardManager` 把环境的 episode reward 用于训练。`RewardDecision.zero_actor_advantage` 是一个声明字段；当前 gateway 只取 `decision.reward`，没有把该标志传进 actor advantage 路径。阅读时不要把它当成已实现的逐样本 advantage 清零。

**评测边界：**当前 `gateway.py` 的 `TaskPlan.transition` 通过 `turns.yml` 的 `success_patterns` 匹配 observation 文本判断完成；没有 `turns.yml` 时使用 `task.md` 和默认步数。它并没有调用完整的 ToolShield 官方 evaluator。`evaluator_evidence` 在此路径也只是 transition 产生的匹配/失败理由。正式安全结论需要独立 ToolShield 评测或更严格的环境证据校验，不能仅凭此 gateway 的 `won`。

## 4. 从真实 GRPO 到 GiRPO 注入

GRPO 基线只使用真实 rollout。`main_ppo_mt_agentrisk.py` 在 `grpo` 模式直接传原 `TrajectoryCollector`；在另外两种模式先读 `FORGET_POOL`，再包一层 `MTAgentRiskForgetCollector`。

`risk_aware_girpo/forget_pool.py` 读取统一 JSONL，只接受 `split=train`，筛选 judge outcome 为 `COMPLETE` 或有较高 `unsafe_intermediate` 的 attempt，并按 `task_id` 保存多条轨迹。`RoundRobinForgetPool` 每次取该任务的下一条。这里的 judge 信息属于输入数据，代码没有再次核实其真实性；建池前应先检查 task split、evaluator 证据和 JEV 冲突。

`risk_aware_girpo/injector.py` 继承 `unlearn/unlearn_inject.py` 的 `ForgetOnlyBaselineExcludedAdvantageCollector`：

1. 真实 group 收集完成后，从 row 的 `task_id` 找到 forget trajectory。
2. 原 GiRPO 代码用保存的 exact prompt/response 重建 token、mask 和 synthetic row。
3. 注入 row 带 `is_forget_injected`、`trajectory_id`、`forget_step_id`、`forget_step_weight`；真实 row 的权重是 1。
4. `ray_trainer.py` 用真实 row 计算 group mean/std 和 GRPO advantage，再给注入 row 负向目标；随后调用 `attach_policy_weights`。

“real-only”是算法要点。如果把注入行也算进组统计，真实样本的 baseline 会被人为拉低，训练目标就变了。原 GiRPO 文件中有多代历史 collector 与长注释；当前 MT-AgentRisk 只用上述继承类，不能因为当前入口没有调用其他类就直接删除，旧实验脚本仍可能引用它们。

### 4.1 JEV 权重何时进入 loss

`scripts/score_jev.py` 先离线调用 `JEVClient`。step 请求只含此前历史、当前 response 与 action；attempt 请求含完整轨迹和最终证据。`jev.py` 对请求做 canonical JSON hash 缓存，并校验 Noul 概率在 `[0,1]`。`jev_girpo` 从步骤读取 `jev_weight`；`girpo` 强制使用 1。同一份 forget pool 因而可用于两者对照。

`risk_aware_girpo/weighting.py` 把 row 的一个标量扩展到该 row 的 advantage token 维度。`dp_actor.py` 和 `megatron_actor.py` 在 PPO policy loss 前计算 `advantages * policy_weights`。这只改变注入步骤的策略梯度贡献，不改 `response_mask` 或 loss 分母；权重 1 与无权重一致，权重 0 则消去该步的直接 policy advantage 项。若开启 KL loss，KL 项仍按原配置工作，因此“权重 0”不等于该 token 完全没有任何梯度。

## 5. 离线数据、基线和脚本怎么选

| 需求 | 入口 | 产物/下一步 |
| --- | --- | --- |
| 将任务元数据固定拆分 | `scripts/build_manifests.py` | train/validation/test manifest |
| 把 ToolShield 轨迹转统一格式 | `scripts/convert_toolshield.py` 或 `run_toolshield_batch.py` | attempt JSONL，需 exact prompt/response/action |
| 给 attempt/step 打 JEV 分 | `scripts/score_jev.py` | 带 judge output 与 step weight 的 JSONL |
| 在线 GRPO 对照 | `scripts/experiments/run_mt_online.sh grpo` | 真实 rollout + veRL checkpoint |
| 在线 GiRPO | 同脚本 `girpo` | 真实 rollout + 等权注入 |
| 在线 JEV-GiRPO | 同脚本 `jev_girpo` | 真实 rollout + JEV 加权注入 |
| 离线 SimNPO | `scripts/experiments/run_simnpo.sh` | 独立 LoRA checkpoint |
| 独立安全评估 | `scripts/run_toolshield_batch.sh` | 与训练分开的 test attempts |

`schema.py` 是统一 attempt/step 的字段检查和 JSONL I/O；`toolshield.py` 要求 exact prompt、model response、parsed action，缺字段即报错，因为凭 state dump 猜文本会破坏 token 重算。`splits.py` 做按任务分组的分层拆分。`simnpo.py` 的 forget 项是长度归一化 response NLL 上的负偏好目标，`train_simnpo.py` 用 Hugging Face/PEFT 训练；它不需要环境交互。旧的 `examples/`、`unlearn/` 含 ALFWorld/WebShop 和 GA、NPO、DPO、RLOO 等实验，属于对照与遗留入口，不能等同 MT-AgentRisk 在线主线。

## 6. 建议的阅读与核验顺序

1. 看 `scripts/experiments/run_mt_online.sh` 的配置覆盖，弄清 group size、模型、数据文件、gateway 地址。
2. 看 `main_ppo_mt_agentrisk.py` 的三种 method 分支和 `RayPPOTrainer` 构造。
3. 沿 `mt_environment.py → gateway.py → openhands_runner.py` 跟一轮 reset/step/close。
4. 沿 `rollout_loop.py → injector.py → unlearn_inject.py` 看真实 row 与注入 row 如何形成。
5. 沿 `ray_trainer.py → weighting.py → dp_actor.py` 看 advantage 和 PPO loss；Megatron 策略则看 `megatron_actor.py`。
6. 最后看 `toolshield.py → score_jev.py → forget_pool.py`，理解离线危险轨迹如何成为注入输入。

先运行 `python -m pytest -q tests/risk_aware_girpo` 检查接合层，再按实现报告配置 gateway 和真实任务包做端到端试运行。单元测试覆盖格式、数值与模拟环境，不证明外部 MCP、OpenHands、JEV 服务和正式 evaluator 已在目标机器上正常工作。
