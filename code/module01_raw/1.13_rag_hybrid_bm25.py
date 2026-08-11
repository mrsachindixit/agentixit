import os
import sys
import numpy as np
from rank_bm25 import BM25Okapi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import embed


CORPUS = [
    {"id": "d1", "text": "Order ORD-2048 was refunded on 2026-05-12 for customer C-77.",
     "meta": {"tenant": "acme",   "type": "invoice",   "year": 2026}},
    {"id": "d2", "text": "Refund policy: 7 days full, 30 days prorated.",
     "meta": {"tenant": "acme",   "type": "policy",    "year": 2026}},
    {"id": "d3", "text": "Customer satisfaction can be improved with proactive outreach.",
     "meta": {"tenant": "acme",   "type": "blog",      "year": 2024}},
    {"id": "d4", "text": "Order ORD-9001 shipped to address on file.",
     "meta": {"tenant": "globex", "type": "invoice",   "year": 2026}},  # other tenant
    {"id": "d5", "text": "Returns are accepted only with original packaging.",
     "meta": {"tenant": "acme",   "type": "policy",    "year": 2025}},
]


def tokenize(s: str) -> list[str]:
    return [t.lower().strip(".,;:!?") for t in s.split() if t]


def cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def metadata_filter(docs, where: dict) -> list[dict]:
    return [d for d in docs if all(d["meta"].get(k) == v for k, v in where.items())]


def bm25_search(query: str, docs: list[dict], k: int) -> list[tuple[str, float]]:
    bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip([d["id"] for d in docs], scores), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def dense_search(query: str, docs: list[dict], k: int) -> list[tuple[str, float]]:
    qv = embed(query)[0]
    doc_vecs = embed([d["text"] for d in docs])
    scores = [(d["id"], cosine(qv, v)) for d, v in zip(docs, doc_vecs)]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]


def rrf(rankings: list[list[tuple[str, float]]], k_const: int = 60) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k_const + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    query = "What did we refund for ORD-2048?"

    candidates = metadata_filter(CORPUS, where={"tenant": "acme"})
    print(f"After tenant filter: {[d['id'] for d in candidates]} "
          f"(blocked: {set(d['id'] for d in CORPUS) - set(d['id'] for d in candidates)})")

    bm25_top = bm25_search(query, candidates, k=4)
    dense_top = dense_search(query, candidates, k=4)
    print(f"\nBM25  top: {bm25_top}")
    print(f"Dense top: {dense_top}")

    fused = rrf([bm25_top, dense_top])
    print(f"\nRRF   top: {fused}")

    print(f"\nFinal context picked: {fused[0][0]}")
