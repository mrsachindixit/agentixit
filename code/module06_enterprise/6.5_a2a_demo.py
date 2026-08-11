import os

import requests

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def ollama_chat(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")

def agent(role_desc: str, message: str) -> str:
    messages = [
        {"role":"system","content": role_desc},
        {"role":"user","content": message}
    ]
    return ollama_chat(messages, temperature=0.2)

if __name__ == "__main__":
    a1_role = "You are Agent A: a planner that decomposes tasks."
    a2_role = "You are Agent B: an executor that writes concrete instructions."

    task = "Build a safe SQL query agent and an explanation of results."
    plan = agent(a1_role, f"Decompose this task: {task}")
    exec_plan = agent(a2_role, f"Turn this plan into exact steps and checks:{plan}")
    final = agent(a1_role, f"Review the executor plan and finalize:{exec_plan}")
    print("== Finalized Plan ==", final)
