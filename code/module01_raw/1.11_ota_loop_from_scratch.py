import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import chat


# --- Tools (the "world") ---------------------------------------------------
def add(a: float, b: float) -> float:        return a + b
def multiply(a: float, b: float) -> float:   return a * b
def lookup_price(item: str) -> float:
    return {"apple": 1.0, "banana": 0.5, "cherry": 3.0}.get(item.lower(), 0.0)

TOOLS = {"add": add, "multiply": multiply, "lookup_price": lookup_price}

TOOL_DESC = """Tools you can call (return JSON only):
  add(a, b)            -> number
  multiply(a, b)       -> number
  lookup_price(item)   -> number  (apple/banana/cherry)
"""


SYSTEM = (
    "You are an agent. On each turn, output JSON with one of:\n"
    '  {"action": "TOOL", "tool": "<name>", "args": {...}}\n'
    '  {"action": "DONE", "answer": "<final answer>"}\n'
    "No prose outside JSON.\n" + TOOL_DESC
)


def think(history: list[dict]) -> dict:
    raw = chat([{"role": "system", "content": SYSTEM}] + history)
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start:end + 1])


def act(decision: dict) -> str:
    tool = TOOLS[decision["tool"]]
    result = tool(**decision["args"])
    return f"Observation: {decision['tool']}({decision['args']}) -> {result}"


def run(goal: str, max_steps: int = 6) -> str:
    history = [{"role": "user", "content": goal}]
    for step in range(max_steps):
        decision = think(history)
        print(f"[step {step}] decision: {decision}")
        if decision.get("action") == "DONE":
            return decision.get("answer", "")
        observation = act(decision)
        print(f"[step {step}] {observation}")
        history.append({"role": "assistant", "content": json.dumps(decision)})
        history.append({"role": "user",      "content": observation})
    return "BUDGET EXHAUSTED"


if __name__ == "__main__":
    goal = "Buy 3 apples and 2 cherries. What is the total cost?"
    answer = run(goal)
    print(f"\nFINAL ANSWER: {answer}")
