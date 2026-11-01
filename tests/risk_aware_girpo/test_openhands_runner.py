from risk_aware_girpo.openhands_runner import _mcp_toml


def test_mcp_config_selects_only_task_dependencies(monkeypatch):
    monkeypatch.setenv("SERVER_HOST", "host")
    value = _mcp_toml(["mcp-filesystem", "unrelated"])
    assert "http://host:9090/sse" in value
    assert "9091" not in value
