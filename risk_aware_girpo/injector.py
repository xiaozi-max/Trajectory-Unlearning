from __future__ import annotations

from typing import Any

import numpy as np

from risk_aware_girpo.forget_pool import RoundRobinForgetPool
from unlearn.unlearn_inject import ForgetOnlyBaselineExcludedAdvantageCollector


class MTAgentRiskForgetCollector(ForgetOnlyBaselineExcludedAdvantageCollector):
    """GiRPO injection keyed by MT-AgentRisk task ID with optional JEV weights."""

    def __init__(
        self,
        *,
        base_collector,
        tokenizer,
        forget_pool: RoundRobinForgetPool,
        config,
        use_jev_weights: bool,
        extra_penalty: float = 0.0,
    ):
        self.pool = forget_pool
        self.use_jev_weights = bool(use_jev_weights)
        self._selected: dict[str, dict[str, Any]] = {}
        super().__init__(
            base_collector=base_collector,
            tokenizer=tokenizer,
            forget_trajectories={},
            config=config,
            extra_penalty=extra_penalty,
            match_key="task_id",
        )

    def _select_for_batch(self, batch) -> dict[str, dict[str, Any]]:
        task_ids = batch.non_tensor_batch.get("task_id")
        step_nums = batch.non_tensor_batch.get("step_num")
        if task_ids is None or step_nums is None:
            return {}
        selected: dict[str, dict[str, Any]] = {}
        for task_id, step_num in zip(task_ids, step_nums, strict=True):
            task_id = str(task_id)
            if int(step_num) == 0 and task_id not in selected:
                entry = self.pool.next(task_id)
                if entry is not None:
                    selected[task_id] = entry
        return selected

    def _inject(self, batch):
        n_real = len(batch)
        self._selected = self._select_for_batch(batch)
        self.forget_trajectories = self._selected
        combined = super()._inject(batch)
        if len(combined) == n_real:
            return combined

        injected = np.asarray(combined.non_tensor_batch["is_forget_injected"], dtype=np.bool_)
        weights = np.ones(len(combined), dtype=np.float32)
        trajectory_ids = np.full(len(combined), "", dtype=object)
        forget_step_ids = np.full(len(combined), -1, dtype=np.int64)
        task_ids = np.asarray(combined.non_tensor_batch["task_id"], dtype=object).copy()
        data_sources = np.asarray(combined.non_tensor_batch["data_source"], dtype=object).copy()

        uid_to_task: dict[object, str] = {}
        uids = combined.non_tensor_batch["uid"]
        step_nums = combined.non_tensor_batch["step_num"]
        for index in range(n_real):
            uid_to_task.setdefault(uids[index], str(task_ids[index]))

        for index in np.nonzero(injected)[0].tolist():
            task_id = uid_to_task[uids[index]]
            entry = self._selected[task_id]
            step_id = int(step_nums[index])
            step = entry["steps"][step_id]
            task_ids[index] = task_id
            trajectory_ids[index] = entry["trajectory_id"]
            forget_step_ids[index] = int(step.get("step_id", step_id))
            weights[index] = float(step["step_weight"]) if self.use_jev_weights else 1.0
            data_sources[index] = "mt_agentrisk"

        combined.non_tensor_batch["task_id"] = task_ids
        combined.non_tensor_batch["trajectory_id"] = trajectory_ids
        combined.non_tensor_batch["forget_step_id"] = forget_step_ids
        combined.non_tensor_batch["forget_step_weight"] = weights
        combined.non_tensor_batch["data_source"] = data_sources
        return combined
