
import os
import requests

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def generate(prompt: str, model: str = DEFAULT_CHAT_MODEL, **options) -> str:
    """
    Wrapper for /api/generate
    """
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def chat(messages, model: str = DEFAULT_CHAT_MODEL, **options) -> str:
    """
    Wrapper for /api/chat
    messages = [{"role":"system|user|assistant", "content":"..."}]
    """
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")

def embed(texts, model: str = DEFAULT_EMBED_MODEL):
    """
    Wrapper for /api/embeddings
    texts: str or List[str]
    Returns: List[List[float]]
    """
    if isinstance(texts, str):
        texts = [texts]
    url = f"{OLLAMA_BASE}/api/embeddings"
    payload = {"model": model, "input": texts}
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    if "embeddings" in data:
        return data["embeddings"]
    elif "data" in data:
        return [item["embedding"] for item in data["data"]]
    else:
        raise ValueError("Unexpected embeddings response format")
