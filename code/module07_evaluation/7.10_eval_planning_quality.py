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


def parse_plan(text: str) -> list[dict]:
    try:
        data = json.loads(text)
        steps = data.get("steps", [])
        return steps if isinstance(steps, list) else []
    except Exception:
        return []


def score_plan(steps: list[dict], required: list[str]) -> dict:
    names = [str(s.get("name", "")).lower() for s in steps]

    # 1. Coverage: how many of the required steps show up at all.
    covered = 0
    for step in required:
        if any(step in name for name in names):
            covered += 1
    coverage = covered / max(len(required), 1)

    # 2. Dependencies declared: every step has a depends_on list (even if empty).
    deps_ok = all(isinstance(s.get("depends_on", []), list) for s in steps)

    # 3. Correct order: the stages must appear load -> split -> embed -> retrieve -> answer.
    pipeline = ["load", "split", "embed", "retrieve", "answer"]
    positions = []
    for stage in pipeline:
        found_at = -1                       # -1 means this stage is missing from the plan
        for i, name in enumerate(names):
            if stage in name:
                found_at = i
                break
        positions.append(found_at)

    all_present = -1 not in positions
    in_order = positions == sorted(positions)   # already-sorted indices => correct order
    order_ok = all_present and in_order

    score = round(0.5 * coverage + 0.25 * int(deps_ok) + 0.25 * int(order_ok), 3)
    return {
        "coverage": round(coverage, 3),
        "deps_ok": int(deps_ok),
        "order_ok": int(order_ok),
        "score": score,
    }


if __name__ == "__main__":
    task = "Build the same local RAG flow students saw: load docs, split chunks, embed, retrieve, answer."
    required_steps = ["load docs", "split", "embed", "retrieve", "answer"]
    user_prompt = (
        f"Task: {task}\n"
        "Return plan as strict JSON: {\"steps\": [{\"name\": str, \"depends_on\": [str]}]}"
    )

    baseline = llm_answer("Create a plan.", user_prompt)
    improved = llm_answer(
        "Create a 5-step plan with explicit dependencies that follows load->split->embed->retrieve->answer.",
        user_prompt,
    )

    baseline_score = score_plan(parse_plan(baseline), required_steps)
    improved_score = score_plan(parse_plan(improved), required_steps)

    print({
        "baseline": {"score": baseline_score.get("score", 0), "details": baseline_score},
        "improved": {"score": improved_score.get("score", 0), "details": improved_score},
    })
