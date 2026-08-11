import json
import os
import random
import sys
import time
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import chat


def lookup_price(item):
    return {"apple": 1.0, "banana": 0.5, "cherry": 3.0}[item.lower()]


_flaky_calls = [0]


def flaky_inventory(item):
    _flaky_calls[0] += 1
    if _flaky_calls[0] <= 2:
        raise TimeoutError("upstream temporarily unavailable")
    return {"apple": 12, "banana": 0, "cherry": 7}.get(item.lower(), 0)


_orders = {}


def place_order(item, qty, idempotency_key):
    if idempotency_key in _orders:
        return _orders[idempotency_key]
    order_id = f"ord-{uuid.uuid4().hex[:8]}"
    _orders[idempotency_key] = order_id
    if random.random() < 0.5:
        raise ConnectionError("response dropped after commit")
    return order_id


TOOLS = {
    "lookup_price": lookup_price,
    "flaky_inventory": flaky_inventory,
    "place_order": place_order,
}

SCHEMAS = {
    "lookup_price": {"item": str},
    "flaky_inventory": {"item": str},
    "place_order": {"item": str, "qty": int, "idempotency_key": str},
}

TOOL_DESC = (
    "Tools you can call (JSON only):\n"
    "  lookup_price(item: str) -> number\n"
    "  flaky_inventory(item: str) -> integer\n"
    "  place_order(item: str, qty: int, idempotency_key: str) -> string\n"
)

SYSTEM = (
    "You are an agent. On each turn output JSON with one of:\n"
    '  {"action":"TOOL","tool":"<name>","args":{...}}\n'
    '  {"action":"DONE","answer":"<final answer>"}\n'
    "No prose outside JSON.\n" + TOOL_DESC
)


def validate_call(decision):
    tool = decision.get("tool")
    if tool not in TOOLS:
        return f"unknown tool '{tool}'. Known: {sorted(TOOLS)}"
    args = decision.get("args", {})
    if not isinstance(args, dict):
        return f"args must be object, got {type(args).__name__}"
    for k, t in SCHEMAS[tool].items():
        if k not in args:
            return f"{tool} missing arg '{k}'"
        if not isinstance(args[k], t):
            return f"{tool} expected {t.__name__} for '{k}', got {type(args[k]).__name__}"
    return None


def execute(decision, max_retries=3):
    tool_name = decision["tool"]
    fn = TOOLS[tool_name]
    args = decision["args"]
    last = ""
    for attempt in range(1, max_retries + 1):
        try:
            result = fn(**args)
            return f"Observation: {tool_name}({args}) -> {result}"
        except (TimeoutError, ConnectionError) as exc:
            last = f"transient: {exc}"
            backoff = 0.05 * (2 ** (attempt - 1))
            print(f"   ! transient on attempt {attempt}: {exc}; sleep {backoff:.2f}s")
            time.sleep(backoff)
        except KeyError as exc:
            return f"Observation: PERMANENT error from {tool_name}: missing {exc}"
        except Exception as exc:
            return f"Observation: PERMANENT error from {tool_name}: {exc}"
    return f"Observation: gave up after {max_retries} attempts ({last})"


def think(history):
    raw = chat([{"role": "system", "content": SYSTEM}] + history)
    s, e = raw.find("{"), raw.rfind("}")
    return json.loads(raw[s:e + 1])


def run(goal, max_steps=8):
    history = [{"role": "user", "content": goal}]
    for step in range(max_steps):
        decision = think(history)
        print(f"[step {step}] decision: {decision}")
        if decision.get("action") == "DONE":
            return decision.get("answer", "")
        err = validate_call(decision)
        obs = f"Observation: VALIDATION FAILED: {err}" if err else execute(decision)
        print(f"[step {step}] {obs}")
        history.append({"role": "assistant", "content": json.dumps(decision)})
        history.append({"role": "user", "content": obs})
    return "BUDGET EXHAUSTED"


def offline_demo():
    print("=== five tool-failure modes ===\n")

    print("1) hallucinated tool name")
    print("  ->", validate_call({"tool": "lookup_pric", "args": {"item": "apple"}}))

    print("\n2) bad arguments")
    print("  ->", validate_call({
        "tool": "place_order",
        "args": {"item": "apple", "qty": "three", "idempotency_key": "k1"},
    }))

    print("\n3) transient runtime error (retries succeed)")
    _flaky_calls[0] = 0
    print("  ->", execute({"tool": "flaky_inventory", "args": {"item": "apple"}}))

    print("\n4) permanent runtime error (no retry)")
    print("  ->", execute({"tool": "lookup_price", "args": {"item": "durian"}}))

    print("\n5) side-effect lost (idempotency replay)")
    key = "demo-key-1"
    for attempt in range(1, 6):
        try:
            order_id = place_order("apple", 3, key)
            print(f"  -> attempt {attempt}: order_id={order_id}")
            replay = place_order("apple", 3, key)
            print(f"  -> replay returned {replay}; same? {order_id == replay}")
            break
        except ConnectionError as exc:
            print(f"  -> attempt {attempt}: dropped ({exc}); retry")


if __name__ == "__main__":
    if "--offline" in sys.argv:
        offline_demo()
    else:
        random.seed(7)
        goal = (
            "Check apples stock, look up price, "
            "place an order for 3 apples with a fresh idempotency key."
        )
        try:
            answer = run(goal)
            print(f"\nFINAL ANSWER: {answer}")
        except Exception as exc:
            print(f"\n(no LLM available: {exc}); running offline demo\n")
            offline_demo()
