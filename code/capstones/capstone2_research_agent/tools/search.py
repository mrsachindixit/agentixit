from typing import List

def search_web(query: str, max_results: int = 5) -> List[dict]:
    """Lightweight placeholder for web/arXiv search.

    Returns a list of dicts: {title, url, snippet}
    Replace with API-backed implementation for production.
    """
    # Placeholder results — in a real project call arXiv, Semantic Scholar, etc.
    sample = [
        {"title": f"Paper matching {query} [{i}]", "url": f"https://example.org/paper/{i}", "snippet": "Short abstract..."}
        for i in range(1, max_results+1)
    ]
    return sample
