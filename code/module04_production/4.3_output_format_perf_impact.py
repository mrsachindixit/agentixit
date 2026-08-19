import os
import time

import requests

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")


def chat(messages, model: str = DEFAULT_CHAT_MODEL, **options) -> str:
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    if options:
        payload["options"] = options
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")


def run(system_prompt: str, question: str) -> tuple[str, float]:
    started = time.perf_counter()
    try:
        answer = chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        ).strip()
    except Exception as exc:
        answer = f"ERROR: {exc}"
    return answer, (time.perf_counter() - started) * 1000


def json_like(text: str) -> int:
    t = text.strip()
    return int(t.startswith("{") and t.endswith("}"))


if __name__ == "__main__":
    question = "From tool stubs, return weather and pincode for Berlin."

    variants = [
        ("no_format", "Answer naturally."),
        ("bullets", "Answer in exactly 2 short bullet points."),
        ("json", "Return strict JSON only: {\"city\": str, \"weather\": str, \"pincode\": str}."),
    ]

    rows = []
    for name, prompt in variants:
        text, latency = run(prompt, question)
        cost_proxy = len(text)
        parse_signal = json_like(text) if name == "json" else int("berlin" in text.lower())
        rows.append({"variant": name, "latency_ms": round(latency, 1), "cost_proxy_chars": cost_proxy, "parse_signal": parse_signal})

    for row in rows:
        print(row)
