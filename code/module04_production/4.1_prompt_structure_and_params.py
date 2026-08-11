import json
import time
import urllib.request


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:latest"


def chat_once(system_prompt, user_prompt, options):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": options,
    }
    started = time.perf_counter()
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["message"]["content"]).strip(), time.perf_counter() - started


query = "Give deployment recommendations for an AI support assistant."

# Weak structure + high randomness: vague role, loose sampling -> rambling, varies each run.
weak_answer, weak_s = chat_once(
    "You are helpful.",
    query,
    {"temperature": 0.9, "top_p": 0.95, "num_predict": 300},
)

# Strong structure + controlled sampling: precise role, tight budget -> short, stable.
strong_answer, strong_s = chat_once(
    "You are a production architect. Output exactly 5 bullets. Max 12 words each.",
    query,
    {"temperature": 0.2, "top_p": 0.8, "num_predict": 120, "repeat_penalty": 1.1},
)

print("=== Weak structure + high randomness ===")
print(f"Latency: {weak_s:.2f}s")
print(weak_answer)

print("\n=== Strong structure + controlled output ===")
print(f"Latency: {strong_s:.2f}s")
print(strong_answer)

print("\nTakeaway: the role text and the sampling options together decide shape, length, and stability.")
