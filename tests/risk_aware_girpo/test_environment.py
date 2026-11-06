from risk_aware_girpo.mt_environment import MTAgentRiskEnvironmentManager, MTAgentRiskVectorEnv


class FakeBackend:
    def reset(self, task_id, seed):
        return {"session_id": f"{task_id}-{seed}", "observation": f"start:{task_id}", "info": {}}

    def step(self, session_id, action):
        return {"observation": f"{session_id}:{action}", "reward": 1.0, "done": True, "won": True}

    def close(self, session_id):
        pass


class InvalidActionBackend(FakeBackend):
    def step(self, session_id, action):
        return {
            "observation": "invalid",
            "reward": 0.0,
            "done": False,
            "won": False,
            "info": {"is_action_valid": False},
        }


def test_reset_step_contract_threads_task_id():
    manager = MTAgentRiskEnvironmentManager(MTAgentRiskVectorEnv(FakeBackend(), ["task-a"], [7]))
    obs, infos = manager.reset()
    assert obs["task_id"] == ["task-a"]
    assert infos[0]["task_id"] == "task-a"
    next_obs, rewards, dones, infos = manager.step(["do"])
    assert next_obs["task_id"] == ["task-a"]
    assert rewards.tolist() == [1.0]
    assert dones.tolist() == [True]
    success = manager.success_evaluator(
        total_infos=[[infos[0]]],
        total_batch_list=[[{"active_masks": True}]],
        episode_rewards=rewards,
        episode_lengths=[1.0],
    )
    assert success["success_rate"].tolist() == [1.0]


def test_step_preserves_gateway_action_validity():
    manager = MTAgentRiskEnvironmentManager(MTAgentRiskVectorEnv(InvalidActionBackend(), ["task-a"], [7]))
    manager.reset()
    _, _, _, infos = manager.step(["bad"])
    assert infos[0]["is_action_valid"] is False
