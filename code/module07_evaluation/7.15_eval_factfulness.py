from utils.ollama_client import chat


def llm(system_prompt: str, user_prompt: str) -> str:
    try:
        return chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def support_ratio(answer: str, must_include: list[str]) -> float:
    text = answer.lower()
    hits = sum(1 for t in must_include if t in text)
    return hits / max(len(must_include), 1)


if __name__ == "__main__":
    context = "Asha is in Engineering. Safe SQL allows only SELECT."
    question = "Who is in Engineering and can DELETE run?"

    baseline = llm("Answer confidently.", f"Context: {context}\nQuestion: {question}")
    improved = llm("Answer only from context. Include Engineering and SELECT-only.", f"Context: {context}\nQuestion: {question}")

    b_score = support_ratio(baseline, ["engineering", "select"])
    i_score = support_ratio(improved, ["engineering", "select"])

    print({
        "baseline": {"score": round(b_score, 4), "details": {"answer": baseline}},
        "improved": {"score": round(i_score, 4), "details": {"answer": improved}},
    })
