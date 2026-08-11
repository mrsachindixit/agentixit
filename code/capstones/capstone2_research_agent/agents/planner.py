import os, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.ollama_client import chat

VALID_ACTIONS = {"search", "ingest", "read", "extract", "synthesize", "summarize"}


def _validate_plan(plan: dict) -> dict:
    steps = plan.get("steps", [])
    issues = []
    if not isinstance(steps, list) or not steps:
        issues.append("steps_missing")
        return {"ok": False, "issues": issues}

    seen = set()
    for step in steps:
        sid = step.get("id")
        action = (step.get("action") or "").strip().lower()
        if sid in seen:
            issues.append(f"duplicate_step_id:{sid}")
        seen.add(sid)
        if action not in VALID_ACTIONS:
            issues.append(f"invalid_action:{action or 'empty'}")
        if not step.get("description"):
            issues.append(f"missing_description:{sid}")
    return {"ok": not issues, "issues": issues}

def plan_research(prompt: str) -> dict:
    """Produce a structured research plan as JSON.

    Returns a dict with keys: title, objectives, steps (ordered list of steps).
    Each step is an object: {id, action, description, tools: []}
    """
    system = (
        "You are a Research Planner assistant. Given a high-level research prompt,"
        " produce a concise JSON object with 'title', 'objectives', and an ordered 'steps' list."
        " Each step should include an 'id', 'action' (search|ingest|read|extract|synthesize), a brief 'description', and 'tools' suggestions."
        " Only output valid JSON."
    )
    messages = [{"role":"system","content":system},{"role":"user","content":prompt}]
    resp = chat(messages, temperature=0.2)
    try:
        plan = json.loads(resp)
    except Exception:
        # fallback simple plan
        plan = {
            "title": prompt[:80],
            "objectives": ["Survey recent methods","Extract key papers","Summarize findings"],
            "steps": [
                {"id": 1, "action": "search", "description": "Search for recent papers and links", "tools": ["search"]},
                {"id": 2, "action": "ingest", "description": "Download and extract PDFs", "tools": ["pdf_ingest"]},
                {"id": 3, "action": "extract", "description": "Extract key findings and methods", "tools": ["summarize"]},
                {"id": 4, "action": "synthesize", "description": "Create a short synthesis", "tools": ["synthesizer"]}
            ]
        }
    check = _validate_plan(plan)
    plan["quality_gate"] = {
        "valid": check["ok"],
        "issues": check["issues"],
        "note": "Planner output must pass structural checks before execution.",
    }
    return plan
