import json
import os

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


VALID_TOOLS = {"user_identity_tool", "horoscope_tool", "weather_tool"}


def llm_answer(system_prompt: str, user_prompt: str) -> str:
    try:
        return chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def parse_steps(text: str) -> list[dict]:
    try:
        data = json.loads(text)
        steps = data.get("steps", [])
        return steps if isinstance(steps, list) else []
    except Exception:
        return []


def score_react(steps: list[dict]) -> dict:
    action_steps = [s for s in steps if isinstance(s, dict) and s.get("action")]
    reasons_before_acting = bool(action_steps) and all(
        str(s.get("thought", "")).strip() for s in action_steps
    )
    valid_tools = bool(action_steps) and all(s.get("action") in VALID_TOOLS for s in action_steps)
    delegated = any(s.get("action") in VALID_TOOLS for s in action_steps)
    terminated = any(isinstance(s, dict) and s.get("final_answer") for s in steps)

    checks = {
        "reasons_before_acting": int(reasons_before_acting),
        "valid_tools": int(valid_tools),
        "delegated": int(delegated),
        "terminated": int(terminated),
    }
    score = round(sum(checks.values()) / len(checks), 3)
    return {"score": score, **checks, "num_steps": len(steps)}


if __name__ == "__main__":
    task = (
        "Supervisor task: user says 'My user ID is user456'. First identify the user, "
        "then if Premium, get their horoscope; otherwise get weather. "
        "Return STRICT JSON only: {\"steps\": [{\"thought\": str, \"action\": str, "
        "\"observation\": str}], with a final element {\"final_answer\": str}}. "
        f"Valid actions: {sorted(VALID_TOOLS)}."
    )

    baseline = llm_answer("Produce a ReAct trace.", task)
    improved = llm_answer(
        "Produce a disciplined ReAct trace: every action MUST be preceded by a thought, "
        "delegate to the correct specialist tool, and end with a final_answer.",
        task,
    )

    baseline_score = score_react(parse_steps(baseline))
    improved_score = score_react(parse_steps(improved))
    print({
        "baseline": {"score": baseline_score["score"], "details": baseline_score},
        "improved": {"score": improved_score["score"], "details": improved_score},
    })
