import os

from utils.ollama_client import chat

JUDGE_A = os.getenv("JUDGE_A_MODEL", "llama3.1:latest")
JUDGE_B = os.getenv("JUDGE_B_MODEL", "lfm2.5-thinking:latest")


def clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


def judge(model: str, question: str, answer: str, required_terms: list[str]) -> int:
    prompt = (
        "Score this answer from 1 to 5 for correctness and relevance. "
        "Return only an integer. "
        f"Question: {question}\nRequired: {', '.join(required_terms)}\nAnswer: {answer}"
    )
    try:
        raw = chat(
            [
                {"role": "system", "content": "You are a strict evaluator."},
                {"role": "user", "content": prompt},
            ],
            model=model,
        ).strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        score = int(digits[:1]) if digits else 1
        return max(1, min(score, 5))
    except Exception:
        return 1


if __name__ == "__main__":
    question = "For user456 in the multi-agent flow, what should happen after membership is fetched?"
    required = ["premium", "horoscope"]

    baseline_answer = "Fetch user details and then maybe provide general guidance."
    improved_answer = "For Premium user456, call horoscope_tool after user_identity_tool and return final action."

    result = {}
    for label, ans in [("baseline", baseline_answer), ("improved", improved_answer)]:
        score_a = judge(JUDGE_A, question, ans, required)
        score_b = judge(JUDGE_B, question, ans, required)
        jury_avg = round((score_a + score_b) / 2, 2)
        disagreement = abs(score_a - score_b)
        result[label] = {
            "score": clamp_0_1(jury_avg / 5),
            "details": {
                "judge_a": score_a,
                "judge_b": score_b,
                "jury_avg": jury_avg,
                "disagreement": disagreement,
            },
        }
    print(result)
