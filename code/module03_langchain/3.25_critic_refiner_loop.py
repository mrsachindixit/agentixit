import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import chat


GENERATOR_SYS = "You are a senior technical writer. Write a clear, concise paragraph."
CRITIC_SYS = (
    "You are a strict editor. Score the draft on three axes (1-5):\n"
    "  clarity, correctness, conciseness\n"
    'Return JSON ONLY: {"clarity":int,"correctness":int,"conciseness":int,"feedback":"..."}'
)
REFINER_SYS = "Rewrite the draft addressing every item in the editor's feedback. Keep it short."

PASS_BAR = 4   # every axis must be >= 4
MAX_ITER = 4


def generate(task: str) -> str:
    return chat([
        {"role": "system", "content": GENERATOR_SYS},
        {"role": "user",   "content": task},
    ]).strip()


def critique(task: str, draft: str) -> dict:
    raw = chat([
        {"role": "system", "content": CRITIC_SYS},
        {"role": "user",   "content": f"Task: {task}\n\nDraft:\n{draft}"},
    ])
    start, end = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[start:end + 1])
    except Exception:
        return {"clarity": 3, "correctness": 3, "conciseness": 3, "feedback": raw[:200]}


def refine(task: str, draft: str, feedback: dict) -> str:
    return chat([
        {"role": "system", "content": REFINER_SYS},
        {"role": "user", "content": (
            f"Task: {task}\n\nDraft:\n{draft}\n\n"
            f"Editor feedback (scores {feedback}):\n{feedback.get('feedback','')}"
        )},
    ]).strip()


def passes_bar(scores: dict) -> bool:
    return all(int(scores.get(k, 0)) >= PASS_BAR for k in ("clarity", "correctness", "conciseness"))


def critic_refiner_loop(task: str) -> tuple[str, list[dict]]:
    history: list[dict] = []
    draft = generate(task)
    for i in range(1, MAX_ITER + 1):
        scores = critique(task, draft)
        history.append({"iter": i, "scores": scores, "draft_preview": draft[:80]})
        print(f"[iter {i}] scores={ {k: scores.get(k) for k in ('clarity','correctness','conciseness')} }")
        if passes_bar(scores):
            print(f"[iter {i}] PASS bar -> stop")
            return draft, history
        draft = refine(task, draft, scores)
    print(f"[iter {MAX_ITER}] BUDGET exhausted -> returning best-effort")
    return draft, history


if __name__ == "__main__":
    task = (
        "Explain in one short paragraph why retrieval-augmented generation reduces "
        "hallucinations, for a CTO audience that knows the basics of LLMs."
    )

    final, history = critic_refiner_loop(task)
    print("\n=== iteration history ===")
    for h in history:
        print(h)
    print("\n=== final draft ===")
    print(final)
