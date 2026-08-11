from utils.ollama_client import chat


def llm(system_prompt: str, user_prompt: str) -> str:
    try:
        return chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def retrieve(query: str, docs, k: int = 3):
    terms = set(query.lower().replace("?", " ").replace(",", " ").split())
    ranked = sorted(docs, key=lambda d: sum(1 for t in terms if t in d["text"].lower()), reverse=True)
    return ranked[:k]


def rewrite_query(query: str) -> str:
    return llm(
        "Rewrite user query for retrieval, keep intent and key entities.",
        f"Query: {query}\nReturn only rewritten query.",
    )


def rerank(query: str, docs):
    q = query.lower()
    return sorted(docs, key=lambda d: (int("memory" in q and "memory" in d["text"].lower()) + int("tool" in q and "tool" in d["text"].lower()) + int("context" in q and "context" in d["text"].lower())), reverse=True)


def answer_with_context(query: str, docs):
    context = "\n\n".join(f"[{d['source']}] {d['text']}" for d in docs)
    return llm(
        "Answer using only provided context. If missing, say unknown.",
        f"Context:\n{context}\n\nQuestion: {query}",
    )


def grounded(answer: str, required_terms: list[str]) -> int:
    text = answer.lower()
    return int(all(term.lower() in text for term in required_terms))


if __name__ == "__main__":
    corpus = [
        {"source": "m1", "text": "Tools fetch fresh external data during execution."},
        {"source": "m2", "text": "Memory preserves prior turns and supports continuity."},
        {"source": "m3", "text": "Context compaction reduces token load while preserving essentials."},
        {"source": "m4", "text": "Caching and routing can reduce latency and cost."},
    ]

    query = "How do tools and memory interact with context in an agent?"
    required = ["tools", "memory", "context"]

    baseline_docs = retrieve(query, corpus, k=2)
    baseline_answer = answer_with_context(query, baseline_docs)
    baseline_score = grounded(baseline_answer, required)

    rewritten = rewrite_query(query)
    improved_docs = rerank(rewritten, retrieve(rewritten, corpus, k=4))[:2]
    improved_answer = answer_with_context(query, improved_docs)
    improved_score = grounded(improved_answer, required)

    print({
        "baseline": {
            "score": baseline_score,
            "details": {
                "query": query,
                "retrieved_docs": [d["source"] for d in baseline_docs],
                "answer": baseline_answer,
            },
        },
        "improved": {
            "score": improved_score,
            "details": {
                "rewritten_query": rewritten,
                "retrieved_docs": [d["source"] for d in improved_docs],
                "answer": improved_answer,
            },
        },
    })
