import json

from utils.ollama_client import chat


def llm_answer(system_prompt: str, user_prompt: str) -> str:
    try:
        return chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def parse_trace(text: str) -> dict:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def score_handoff(trace: dict) -> dict:
    handoffs = trace.get("handoffs", [])
    if not isinstance(handoffs, list):
        handoffs = []
    pairs = [
        (str(h.get("from", "")).lower(), str(h.get("to", "")).lower())
        for h in handoffs
        if isinstance(h, dict)
    ]
    ping_pong = sum(1 for i in range(1, len(pairs)) if pairs[i] == (pairs[i - 1][1], pairs[i - 1][0]))
    delegated_once = int(any(frm == "master_agent" and to in {"user_identity_tool", "horoscope_tool", "weather_tool"} for frm, to in pairs))
    premium_route_ok = int(("master_agent", "user_identity_tool") in pairs and ("master_agent", "horoscope_tool") in pairs)
    terminated = int(str(trace.get("final_status", "")).lower() in {"done", "success", "partial_success"})
    score = round((0.35 * delegated_once) + (0.35 * premium_route_ok) + (0.2 * terminated) + (0.1 * int(ping_pong == 0)), 3)
    return {
        "delegated_once": delegated_once,
        "premium_route_ok": premium_route_ok,
        "terminated": terminated,
        "ping_pong_loops": ping_pong,
        "score": score,
    }


if __name__ == "__main__":
    task = (
        "Task: user says 'My user ID is user456'. Route through master_agent and sub-agent tools. "
        "Return strict JSON: {\"handoffs\": [{\"from\": str, \"to\": str, \"reason\": str}], \"final_status\": str}."
    )

    baseline = llm_answer("Create a handoff trace.", task)
    improved = llm_answer(
        "Create an efficient handoff trace with minimal loops, premium-aware routing, and clear termination.",
        task,
    )

    baseline_score = score_handoff(parse_trace(baseline))
    improved_score = score_handoff(parse_trace(improved))

    print({
        "baseline": {"score": baseline_score.get("score", 0), "details": baseline_score},
        "improved": {"score": improved_score.get("score", 0), "details": improved_score},
    })
