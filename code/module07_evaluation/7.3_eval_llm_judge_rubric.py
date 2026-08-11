import json

from utils.ollama_client import chat


def llm_answer(system_prompt: str, user_prompt: str) -> str:
    try:
        return chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def llm_judge(question: str, answer: str, must_include: list[str]) -> dict:
    judge_prompt = (
        "Score this answer from 1 to 5 for relevance, correctness, and clarity. "
        "Return strict JSON with keys relevance, correctness, clarity, total. "
        f"Question: {question}\nRequired terms: {', '.join(must_include)}\nAnswer: {answer}"
    )
    raw = llm_answer("You are a strict evaluator.", judge_prompt)
    try:
        return json.loads(raw)
    except Exception:
        return {"relevance": 1, "correctness": 1, "clarity": 1, "total": 3, "raw": raw}


def clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    question = "In the multi-agent sample, user456 has Premium membership. What action should happen after membership is fetched?"
    required = ["premium", "horoscope"]

    baseline = llm_answer("Answer casually.", question)
    improved = llm_answer("Answer in one sentence and include Premium and horoscope.", question)

    baseline_score = llm_judge(question, baseline, required)
    improved_score = llm_judge(question, improved, required)

    print({
        "baseline": {
            "score": clamp_0_1(float(baseline_score.get("total", 0)) / 15),
            "details": baseline_score,
        },
        "improved": {
            "score": clamp_0_1(float(improved_score.get("total", 0)) / 15),
            "details": improved_score,
        },
    })
