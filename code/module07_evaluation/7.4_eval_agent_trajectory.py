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


def parse_tool_calls(text: str) -> list[str]:
    try:
        data = json.loads(text)
        calls = data.get("tool_calls", [])
        return calls if isinstance(calls, list) else []
    except Exception:
        return []


def eval_trajectory(tool_calls: list[str], expected_tool: str, max_steps: int) -> dict:
    used_expected_tool = expected_tool in tool_calls
    step_ok = len(tool_calls) <= max_steps
    return {
        "tool_calls": tool_calls,
        "used_expected_tool": used_expected_tool,
        "step_ok": step_ok,
        "pass": int(used_expected_tool and step_ok),
    }


if __name__ == "__main__":
    task = "User says: My user ID is user456. Use the supervisor flow to complete the correct next action."
    tools = "Available tools: user_identity_tool, horoscope_tool, weather_tool"

    baseline = llm_answer(
        "Plan however you want.",
        f"{task}\n{tools}\nReturn anything you like.",
    )
    improved = llm_answer(
        "Return strict JSON only: {\"tool_calls\": [...], \"final_answer\": \"...\"}. Max 3 tools.",
        f"{task}\n{tools}\nUse user identity first. For Premium membership route to horoscope flow.",
    )

    baseline_eval = eval_trajectory(parse_tool_calls(baseline), expected_tool="horoscope_tool", max_steps=3)
    improved_eval = eval_trajectory(parse_tool_calls(improved), expected_tool="horoscope_tool", max_steps=3)
    print({
        "baseline": {"score": baseline_eval.get("pass", 0), "details": baseline_eval},
        "improved": {"score": improved_eval.get("pass", 0), "details": improved_eval},
    })
