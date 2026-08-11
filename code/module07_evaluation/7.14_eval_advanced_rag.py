from utils.ollama_client import chat


def llm(system_prompt: str, user_prompt: str) -> str:
    try:
        return chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def score(answer: str, required_terms: list[str]) -> float:
    text = answer.lower()
    hits = sum(1 for t in required_terms if t in text)
    return hits / max(len(required_terms), 1)


if __name__ == "__main__":
    query = "How do tools and memory interact with context in an agent?"
    context_baseline = "Tools fetch data."
    context_improved = "Tools fetch data, memory preserves prior turns, context controls relevance."

    baseline = llm("Answer from context.", f"Context: {context_baseline}\nQuestion: {query}")
    improved = llm("Answer from context only and include tools, memory, context.", f"Context: {context_improved}\nQuestion: {query}")

    b_score = score(baseline, ["tools", "memory", "context"])
    i_score = score(improved, ["tools", "memory", "context"])

    print({
        "baseline": {"score": round(b_score, 4), "details": {"answer": baseline}},
        "improved": {"score": round(i_score, 4), "details": {"answer": improved}},
    })
