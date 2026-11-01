from risk_aware_girpo.mt_environment import MTAgentRiskEnvironmentManager, MTAgentRiskVectorEnv


class FakeBackend:
    def reset(self, task_id, seed):
        return {"session_id": f"{task_id}-{seed}", "observation": f"start:{task_id}", "info": {}}

    def step(self, session_id, action):
        return {"observation": f"{session_id}:{action}", "reward": 1.0, "done": True, "won": True}

    def close(self, session_id):
        pass


def test_reset_step_contract_threads_task_id():
    manager = MTAgentRiskEnvironmentManager(MTAgentRiskVectorEnv(FakeBackend(), ["task-a"], [7]))
    obs, infos = manager.reset()
    assert obs["task_id"] == ["task-a"]
    assert infos[0]["task_id"] == "task-a"
    next_obs, rewards, dones, infos = manager.step(["do"])
    assert next_obs["task_id"] == ["task-a"]
    assert rewards.tolist() == [1.0]
    assert dones.tolist() == [True]
