from utils.ollama_client import chat


def llm_answer(system_prompt, user_prompt):
    try:
        return chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def grounded(answer, required_terms):
    text = answer.lower()
    return int(all(term.lower() in text for term in required_terms))


if __name__ == "__main__":
    question = "How do tools and memory interplay in agents?"
    retrieved_context = (
        "Tools fetch fresh external data when needed. "
        "Memory preserves prior conversation context across turns."
    )
    required_terms = ["tools", "memory", "context"]

    baseline = llm_answer("Answer from your general knowledge.", question)
    improved = llm_answer(
        "Answer using only the provided context and include all required terms.",
        f"Context: {retrieved_context}\nQuestion: {question}\nRequired terms: {', '.join(required_terms)}",
    )

    baseline_score = grounded(baseline, required_terms)
    improved_score = grounded(improved, required_terms)
    print({
        "baseline": {"score": baseline_score, "details": {"grounded": baseline_score}},
        "improved": {"score": improved_score, "details": {"grounded": improved_score}},
    })
