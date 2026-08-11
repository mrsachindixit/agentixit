import json
import uuid


ALLOWED_INTENTS = {"summarize_pdf", "extract_tables", "translate"}
ALLOWED_CONSTRAINTS = {
    "read_only",
    "no_external_calls",
    "no_pii_in_output",
    "max_runtime_30s",
}
REQUIRED_FIELDS = ["from", "to", "task_id", "intent", "inputs",
                   "constraints", "expected_artifact"]


def validate_envelope(env):
    missing = [f for f in REQUIRED_FIELDS if f not in env]
    if missing:
        return f"missing fields: {missing}"
    if not isinstance(env["task_id"], str) or len(env["task_id"]) < 3:
        return "task_id must be a string of length >= 3"
    if env["intent"] not in ALLOWED_INTENTS:
        return f"intent '{env['intent']}' not in {sorted(ALLOWED_INTENTS)}"
    if not isinstance(env["inputs"], dict):
        return "inputs must be an object"
    if not isinstance(env["constraints"], list):
        return "constraints must be a list"
    bad = [c for c in env["constraints"] if c not in ALLOWED_CONSTRAINTS]
    if bad:
        return f"unknown constraints {bad}; allowed: {sorted(ALLOWED_CONSTRAINTS)}"
    if not isinstance(env["expected_artifact"], str) or not env["expected_artifact"]:
        return "expected_artifact must be a non-empty string"
    return None


def planner(user_goal):
    if "summarize" in user_goal.lower() or "summary" in user_goal.lower():
        env = {
            "from": "planner",
            "to": "executor",
            "task_id": f"t-{uuid.uuid4().hex[:6]}",
            "intent": "summarize_pdf",
            "inputs": {"path": "/docs/spec.pdf", "max_tokens": 800},
            "constraints": ["read_only", "no_external_calls"],
            "expected_artifact": "summary.md",
        }
        err = validate_envelope(env)
        if err:
            raise ValueError(f"planner produced bad envelope: {err}")
        return env
    raise ValueError(f"planner cannot map goal: {user_goal!r}")


def executor(env):
    err = validate_envelope(env)
    if err:
        raise ValueError(f"executor refusing bad envelope: {err}")
    if env["intent"] != "summarize_pdf":
        raise RuntimeError(f"executor does not implement intent '{env['intent']}'")
    return {
        "artifact_name": env["expected_artifact"],
        "task_id": env["task_id"],
        "body": f"(pretend summary of {env['inputs']['path']})",
    }


def demo_happy_path():
    print("--- happy path: planner -> envelope -> executor ---")
    env = planner("Please summarize the spec document")
    on_wire = json.dumps(env)
    reborn = json.loads(on_wire)
    assert reborn == env, "envelope did not round-trip"
    print("envelope:", json.dumps(env, indent=2))
    result = executor(env)
    assert result["task_id"] == env["task_id"]
    print("result:", json.dumps(result, indent=2))
    print("OK\n")


def demo_caught_failures():
    print("--- broken envelopes the validator catches ---")
    bad = [
        ("invented intent", {
            "from": "planner", "to": "executor", "task_id": "t-bad1",
            "intent": "delete_database", "inputs": {}, "constraints": [],
            "expected_artifact": "ok.md",
        }),
        ("unknown constraint", {
            "from": "planner", "to": "executor", "task_id": "t-bad2",
            "intent": "summarize_pdf", "inputs": {"path": "/x.pdf"},
            "constraints": ["YOLO_mode"], "expected_artifact": "s.md",
        }),
        ("missing expected_artifact", {
            "from": "planner", "to": "executor", "task_id": "t-bad3",
            "intent": "summarize_pdf", "inputs": {"path": "/x.pdf"},
            "constraints": [],
        }),
        ("empty task_id", {
            "from": "planner", "to": "executor", "task_id": "",
            "intent": "summarize_pdf", "inputs": {}, "constraints": [],
            "expected_artifact": "s.md",
        }),
    ]
    for label, env in bad:
        err = validate_envelope(env)
        if err:
            print(f"OK [{label}] caught: {err}")
        else:
            print(f"!! [{label}] UNEXPECTEDLY PASSED")
    print()


def demo_replay():
    print("--- replay a handoff from its serialized form ---")
    wire = ('{"from":"planner","to":"executor","task_id":"t-replay",'
            '"intent":"summarize_pdf","inputs":{"path":"/docs/x.pdf",'
            '"max_tokens":400},"constraints":["read_only"],'
            '"expected_artifact":"summary.md"}')
    env = json.loads(wire)
    result = executor(env)
    print("replayed:", json.dumps(result, indent=2))
    print("OK\n")


if __name__ == "__main__":
    demo_happy_path()
    demo_caught_failures()
    demo_replay()
    print("All checks passed.")
