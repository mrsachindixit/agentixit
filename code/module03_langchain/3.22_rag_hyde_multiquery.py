import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import chat, embed


CORPUS = [
    "RAG combines retrieval with generation: fetch relevant docs, then answer using them.",
    "Vector embeddings let you find semantically similar text by cosine distance.",
    "Hallucinations happen when the model invents facts not in the retrieved context.",
    "Reranking with a cross-encoder improves precision over bi-encoder retrieval.",
    "Hybrid search combines BM25 lexical scores with dense vector scores.",
    "Agents use tools to fetch fresh data the model could not know.",
    "Prompt caching reduces latency and cost for long static prefixes.",
]


def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def retrieve(query: str, k: int = 3) -> list[tuple[str, float]]:
    qv = embed(query)[0]
    dv = embed(CORPUS)
    scored = [(doc, cosine(qv, v)) for doc, v in zip(CORPUS, dv)]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]


def hyde(query: str) -> str:
    return chat([
        {"role": "system", "content": "Write a concise one-paragraph answer as if you were a textbook. Do not refuse."},
        {"role": "user", "content": query},
    ]).strip()


def multi_query(query: str, n: int = 3) -> list[str]:
    raw = chat([
        {"role": "system", "content": f"Generate {n} diverse reformulations of the user query, one per line, no numbering."},
        {"role": "user", "content": query},
    ])
    return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()][:n]


def decompose(query: str) -> list[str]:
    raw = chat([
        {"role": "system", "content": "Split this question into 2-4 atomic sub-questions, one per line, no numbering."},
        {"role": "user", "content": query},
    ])
    return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]


def step_back(query: str) -> str:
    return chat([
        {"role": "system", "content": "Rewrite as a more general, conceptual question. Return only the question."},
        {"role": "user", "content": query},
    ]).strip()


if __name__ == "__main__":
    query = "Why does my RAG system make stuff up even when documents look relevant?"

    print(f"ORIGINAL: {query}")
    baseline = retrieve(query, k=2)
    print(f"  baseline top: {[d for d, _ in baseline]}\n")

    hypo = hyde(query)
    print(f"HYDE hypothetical answer: {hypo[:120]}...")
    print(f"  retrieved: {[d for d, _ in retrieve(hypo, k=2)]}\n")

    variants = multi_query(query)
    print("MULTI-QUERY variants:")
    union: dict[str, float] = {}
    for v in variants:
        print(f"  - {v}")
        for doc, score in retrieve(v, k=2):
            union[doc] = max(union.get(doc, 0.0), score)
    union_top = sorted(union.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  union top:  {[d for d, _ in union_top]}\n")

    subs = decompose(query)
    print("DECOMPOSITION sub-questions:")
    for s in subs:
        print(f"  - {s}  ->  {[d for d, _ in retrieve(s, k=1)]}")

    sb = step_back(query)
    print(f"\nSTEP-BACK: {sb}")
    print(f"  retrieved: {[d for d, _ in retrieve(sb, k=2)]}")

    print("\nProduction tip: combine multi-query + step-back for breadth, then cross-encoder rerank for precision.")
