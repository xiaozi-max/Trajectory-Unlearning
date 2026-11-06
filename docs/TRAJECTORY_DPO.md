# MT-AgentRisk Trajectory DPO

输入是 train split 的统一 attempt JSONL，包含每个 decision step 的 exact `prompt`、`model_response`、`action`、`observation`，以及经过核实的 `judge_output.outcome` 和 `evaluator_evidence`。

另建人工审核的 pair manifest JSONL，每行例如：

```json
{"task_id":"task-1","chosen_attempt_id":"safe-1","rejected_attempt_id":"unsafe-1","reviewed":true}
```

两条 attempt 必须属于同一 harmful task。chosen 只能是经核实的 `REJECT`，rejected 只能是经核实的 `COMPLETE`；`FAILED` 不作为安全样本。Pair 必须来自 train split。原始 attempt 保留完整 observations 供审核；训练时每一步使用其记录的 exact prompt，因此 observation 通过下一步 prompt 进入条件上下文。若记录的 prompt 未包含此前事件，需先修正采集流程。

```bash
export DPO_ATTEMPTS=/path/to/train_attempts.jsonl
export DPO_PAIRS=/path/to/reviewed_pairs.jsonl
bash scripts/run_trajectory_dpo.sh --model Qwen/Qwen2.5-Coder-3B-Instruct
```

每条 trajectory 的 score 是各步实际 response token 的 conditional log probability 之和除以 response token 总数；prompt 与 padding token 均被 mask。每步只截断 prompt 左侧，response 超长时直接报错；不会在中间 step 人为添加 EOS。Reference 为同一 base model 关闭 LoRA adapter 的输出。训练产物是 LoRA adapter，需按与其他组相同的 held-out tasks/seeds 独立评测。此实现是针对本项目的 trajectory-level DPO 设计，不声称复现所引论文的完整算法。
