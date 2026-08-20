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
 
# System prompt forces the base Llama 3.1 model to generate a reasoning trace
system_instruction = (
    "You must think step-by-step before answering. "
    "Output your entire internal reasoning process inside <think>...</think> tags. "
    "After the closing tag, provide your final concise answer."
)
 
response = chat(
    messages=[
        {'role': 'system', 'content': system_instruction},
        {'role': 'user', 'content': 'If a store sells 4 apples for $3 and 3 oranges for $2, which is cheaper per piece?'}
    ],
    model='llama3.1:latest',  # Standard Llama 3.1 model
)
 
print(response)