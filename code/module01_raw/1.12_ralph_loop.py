import os
import re
import subprocess
import sys
import textwrap
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import chat


BUGGY_CODE = textwrap.dedent("""
    def add(a, b):
        return a - b   # bug: should be +

    def mul(a, b):
        return a + b   # bug: should be *
""").strip()

TESTS = textwrap.dedent("""
    from solution import add, mul
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert mul(2, 3) == 6
    assert mul(0, 99) == 0
    print("ALL TESTS PASSED")
""").strip()


SYSTEM = (
    "You are a code-fixing agent. Output ONLY a fenced ```python code block "
    "containing the FULL corrected file contents. No commentary."
)


def extract_code(reply: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.DOTALL)
    return (m.group(1) if m else reply).strip()


def run_tests(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        (open(os.path.join(d, "solution.py"), "w")).write(code)
        (open(os.path.join(d, "test.py"), "w")).write(TESTS)
        proc = subprocess.run(
            [sys.executable, "test.py"],
            cwd=d, capture_output=True, text=True, timeout=10,
        )
        passed = proc.returncode == 0 and "ALL TESTS PASSED" in proc.stdout
        return passed, (proc.stdout + proc.stderr).strip()


def ralph_step(code: str, last_error: str) -> str:
    prompt = (
        f"The code below fails its tests with this error:\n\n{last_error}\n\n"
        f"Current code:\n```python\n{code}\n```\n\nReturn the fixed file."
    )
    reply = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": prompt},
    ])
    return extract_code(reply)


if __name__ == "__main__":
    code = BUGGY_CODE
    max_iterations = 5

    for i in range(1, max_iterations + 1):
        passed, output = run_tests(code)
        print(f"--- iteration {i} ---")
        print(f"tests: {'PASS' if passed else 'FAIL'}")
        if passed:
            print("Ralph terminated successfully.")
            break
        print(f"error tail: {output.splitlines()[-1] if output else '(none)'}")
        code = ralph_step(code, output)

    print("\nFinal code:")
    print(code)
