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
TARGET = ROOT / "module01_raw/1.1_hello_llm.py"


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

import socket
import sys

sys.path.append(str(ROOT))
from utils.ollama_client import generate


def _ollama_available(host: str = "localhost", port: int = 11434) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not running on localhost:11434")
def test_prompt_regression_agent_keyword() -> None:
    output = generate("Define an AI agent in one concise sentence.", temperature=0)
    assert "agent" in output.lower()
