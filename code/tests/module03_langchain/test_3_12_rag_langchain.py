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
TARGET = ROOT / "module03_langchain/3.12_rag_langchain.py"


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
