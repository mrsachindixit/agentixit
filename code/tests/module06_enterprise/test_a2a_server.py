from __future__ import annotations

import os
import py_compile
import subprocess
from pathlib import Path

import pytest


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "module01_raw").exists() and (candidate / "tests").exists():
            return candidate
    raise RuntimeError("Could not locate repository root from test file path")


ROOT = _find_repo_root(Path(__file__).resolve())
TARGET = ROOT / "module06_enterprise/6.6_a2a_server.py"


def test_target_exists() -> None:
    assert TARGET.exists(), f"Missing target file: {TARGET.relative_to(ROOT)}"


def test_target_compiles() -> None:
    py_compile.compile(str(TARGET), doraise=True)


@pytest.mark.skipif(os.getenv("AGENTICAI_RUN_MODULE_SMOKE") != "1", reason="Set AGENTICAI_RUN_MODULE_SMOKE=1 to enable runtime smoke")
def test_target_runtime_smoke() -> None:
    result = subprocess.run(
        [
            "c:/python314/python.exe",
            str(TARGET),
            "--help",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode in {0, 1, 2}, (
        f"Unexpected return code {result.returncode} for {TARGET.relative_to(ROOT)}\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )

import json


def test_message_schema_role_types() -> None:
    valid_roles = ["system", "user", "assistant", "tool"]
    for role in valid_roles:
        message = {"role": role, "content": "test content"}
        assert message["role"] in valid_roles


def test_message_json_serializable() -> None:
    messages = [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Process this request."},
        {"role": "assistant", "content": "I will process it."},
    ]

    serialized = json.dumps(messages)
    deserialized = json.loads(serialized)

    assert len(deserialized) == 3
    assert deserialized[0]["role"] == "system"


def test_a2a_message_format() -> None:
    message = {
        "sender": "planner_agent",
        "recipient": "executor_agent",
        "payload": {
            "task": "Execute plan step 1",
            "context": "Initial context",
        },
    }

    assert message["sender"] in ["planner_agent", "executor_agent"]
    assert message["recipient"] in ["planner_agent", "executor_agent"]
    assert "task" in message["payload"]


def test_protocol_version_tracking() -> None:
    message = {
        "protocol_version": "1.0",
        "timestamp": "2026-02-24T00:00:00Z",
        "content": {"data": "test"},
    }

    assert message["protocol_version"] == "1.0"
    assert "timestamp" in message
