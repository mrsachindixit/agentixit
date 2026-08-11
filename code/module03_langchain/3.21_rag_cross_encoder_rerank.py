from sentence_transformers import SentenceTransformer, CrossEncoder

bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

CORPUS = [
    "Tools fetch fresh data during agent execution and may have side effects.",
    "Memory preserves prior conversation turns and supports continuity across sessions.",
    "Context compaction summarises old turns to free up the context window.",
    "Caching identical prompts can reduce both latency and cost.",
    "An agent loop observes, thinks, and acts repeatedly until done.",
    "Hallucinations occur when an LLM produces content not grounded in evidence.",
    "Vector embeddings encode semantic meaning into fixed-length numerical arrays.",
    "BM25 is a lexical retrieval algorithm based on term frequency.",
    "Cross-encoders score query-document pairs jointly for higher accuracy.",
    "Multi-tenant systems must filter by tenant_id before retrieval to prevent leakage.",
]

QUERY = "How does an agent decide when to act and how does it remember context?"


def stage1_biencoder(query: str, docs: list[str], k: int) -> list[tuple[int, float]]:
    qv = bi_encoder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    dv = bi_encoder.encode(docs,  convert_to_tensor=True, normalize_embeddings=True)
    scores = (dv @ qv).tolist()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def stage2_crossencoder(query: str, docs: list[str], candidate_idx: list[int], n: int) -> list[tuple[int, float]]:
    pairs = [(query, docs[i]) for i in candidate_idx]
    scores = cross_encoder.predict(pairs).tolist()
    ranked = sorted(zip(candidate_idx, scores), key=lambda x: x[1], reverse=True)
    return ranked[:n]


if __name__ == "__main__":
    bi_top = stage1_biencoder(QUERY, CORPUS, k=5)
    print("Stage 1 — Bi-encoder top 5 (fast, broad):")
    for idx, score in bi_top:
        print(f"  {score:+.3f}  {CORPUS[idx]}")

    candidates = [i for i, _ in bi_top]
    cross_top = stage2_crossencoder(QUERY, CORPUS, candidates, n=2)
    print("\nStage 2 — Cross-encoder top 2 (slow, precise):")
    for idx, score in cross_top:
        print(f"  {score:+.3f}  {CORPUS[idx]}")

    print("\nNote: bi-encoder cost is ~O(N) with index; cross-encoder is ~O(K) at query time.")
    print("So K must stay small (5-50). Never run cross-encoder on the whole corpus.")
