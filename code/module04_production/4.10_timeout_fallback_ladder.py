import json
import urllib.error
import urllib.request


PRIMARY_URL = "http://localhost:11434/api/chat"
PRIMARY_MODEL = "lfm2.5-thinking:latest"
FALLBACK_MODEL = "llama3.1:latest"


def call_ollama(model: str, prompt: str, timeout_seconds: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    req = urllib.request.Request(
        PRIMARY_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["message"]["content"])


def answer_with_ladder(prompt: str) -> str:
    try:
        return "[PRIMARY] " + call_ollama(PRIMARY_MODEL, prompt, timeout_seconds=3)
    except Exception:
        try:
            return "[FALLBACK] " + call_ollama(FALLBACK_MODEL, prompt, timeout_seconds=4)
        except Exception:
            return "[DEGRADED] Service is busy. Please retry in a few seconds."


if __name__ == "__main__":
    query = "Give 4 deployment recommendations for a production AI support bot."
    print(answer_with_ladder(query))
