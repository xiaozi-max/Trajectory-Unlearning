from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


class ToolShieldRunner:
    """Execute a small serialized action protocol in ToolShield's pinned OpenHands runtime."""

    def __init__(self, task_path: Path, session_dir: Path, seed: int):
        # ToolShield constructs an unused OpenAI client at import time. The actual
        # OpenHands LLM settings below come from AGENT_CONFIG_FILE.
        os.environ.setdefault("OPENAI_API_KEY", "toolshield-gateway-unused")
        from agentrisk import run_eval
        from openhands.core.config import get_llm_config_arg
        from openhands.core.main import create_runtime
        from openhands.utils.async_utils import call_async_from_sync

        config_file = Path(os.environ["AGENT_CONFIG_FILE"])
        dependencies = run_eval.read_host_dependencies(str(task_path))
        merged_config = session_dir / "run_config.toml"
        merged_config.write_text(
            config_file.read_text(encoding="utf-8") + _mcp_toml(dependencies), encoding="utf-8"
        )
        llm_config = get_llm_config_arg("agent", toml_file=str(merged_config))
        if llm_config is None:
            raise ValueError(f"missing [llm.agent] in {config_file}")

        workspace = session_dir / "workspace"
        workspace.mkdir()
        for source_name in ("workspace", "mcp_fs"):
            source = task_path / source_name
            if source.is_dir():
                shutil.copytree(source, workspace, dirs_exist_ok=True)

        run_eval.host_dependencies = dependencies
        config = run_eval.get_config(
            str(task_path), task_path.name, str(session_dir), llm_config, config_file=str(merged_config)
        )
        # Terminal-only tasks do not need the large Playwright/Chromium runtime layer.
        config.enable_browser = "mcp-playwright" in dependencies
        config.workspace_mount_path = str(workspace)
        config.workspace_mount_path_in_sandbox = "/workspace"
        config.file_store_path = str(session_dir / "file_store")
        config.sandbox.volumes = f"{workspace}:/workspace"
        config.save_trajectory_path = str(session_dir / "trajectory.json")
        config.sandbox.runtime_startup_env_vars["MT_AGENTRISK_SEED"] = str(seed)

        self.runtime = create_runtime(config, sid=session_dir.name)
        call_async_from_sync(self.runtime.connect)
        _copy_task_files(self.runtime, task_path)

    def execute(self, action: dict[str, Any]) -> str:
        from openhands.events.serialization.action import action_from_dict

        observation = self.runtime.run_action(action_from_dict(action))
        return str(getattr(observation, "content", None) or observation)

    def close(self) -> None:
        self.runtime.close()


def _mcp_toml(dependencies: list[str]) -> str:
    host = os.environ.get("SERVER_HOST", "localhost")
    registry = {
        "mcp-filesystem": f"http://{host}:{os.environ.get('MCP_FILESYSTEM_PORT', '9090')}/sse",
        "mcp-postgres": f"http://{host}:{os.environ.get('MCP_POSTGRES_PORT', '9091')}/sse",
        "mcp-playwright": f"http://{host}:{os.environ.get('MCP_PLAYWRIGHT_PORT', '9092')}/sse",
        "mcp-notion": f"http://{host}:{os.environ.get('MCP_NOTION_PORT', '9097')}/sse",
    }
    sse = [registry[name] for name in dependencies if name in registry]
    shttp = [os.environ["MCP_GMAIL_URL"]] if "mcp-gmail" in dependencies and os.environ.get("MCP_GMAIL_URL") else []
    if not sse and not shttp:
        return ""
    lines = ["", "[mcp]"]
    if sse:
        lines.append(f"sse_servers = {sse!r}")
    if shttp:
        lines.append(f"shttp_servers = {shttp!r}")
    return "\n".join(lines) + "\n"


def _copy_task_files(runtime, task_path: Path) -> None:
    copies = [
        (task_path / "utils", "/utils/", True),
        (task_path / "scenarios.json", "/npc/", False),
        (task_path / "task.md", "/instruction/", False),
    ]
    for source, destination, recursive in copies:
        if source.exists():
            runtime.copy_to(host_src=str(source), sandbox_dest=destination, recursive=recursive)
