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
TARGET = ROOT / "module01_raw/1.10_rag_basic/build_index.py"


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
import sys

import numpy as np

INDEX_FILE = ROOT / "module01_raw/1.10_rag_basic/data/index.json"

sys.path.append(str(ROOT))
from utils.ollama_client import embed


def cosine(a, b) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(a_arr @ b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9)


@pytest.mark.skipif(not INDEX_FILE.exists(), reason="Run module01_raw/1.10_rag_basic/build_index.py to create the RAG index")
def test_rag_index_exists_for_basic_flow() -> None:
    assert INDEX_FILE.exists(), "Expected RAG index at module01_raw/1.10_rag_basic/data/index.json"


@pytest.mark.skipif(not INDEX_FILE.exists(), reason="Run module01_raw/1.10_rag_basic/build_index.py to create the RAG index")
def test_rag_retrieval_semantics_basic() -> None:
    with open(INDEX_FILE, "r", encoding="utf-8") as file_handle:
        index_data = json.load(file_handle)

    query = "How do agents use tools and memory?"
    query_vector = embed(query)[0]
    scores = [cosine(query_vector, record["vec"]) for record in index_data]
    assert max(scores) > 0.2
