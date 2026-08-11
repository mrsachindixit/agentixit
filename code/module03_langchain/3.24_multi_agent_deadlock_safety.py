import time
from collections import deque


def buggy_router(state: dict) -> str:
    state["log"].append("router -> specialist")
    return "specialist"

def buggy_specialist(state: dict) -> str:
    state["log"].append("specialist -> router")
    return "router"


AGENTS = {"router": buggy_router, "specialist": buggy_specialist}


def run_unsafe(start: str, state: dict, hard_iter_cap: int = 200) -> str:
    current = start
    for _ in range(hard_iter_cap):
        next_agent = AGENTS[current](state)
        current = next_agent
    return "BUDGET BURNED"


def make_breaker(max_hops: int = 12, cycle_window: int = 6, wallclock_seconds: float = 5.0) -> dict:
    return {
        "max_hops": max_hops,
        "deadline": time.monotonic() + wallclock_seconds,
        "history": deque(maxlen=cycle_window),
        "hops": 0,
    }


def breaker_check(breaker: dict, next_agent: str) -> tuple[bool, str]:
    breaker["hops"] += 1
    if breaker["hops"] > breaker["max_hops"]:
        return False, f"hop_limit({breaker['max_hops']})"
    if time.monotonic() > breaker["deadline"]:
        return False, "watchdog_timeout"
    breaker["history"].append(next_agent)
    if breaker["history"].count(next_agent) >= 3:
        return False, f"cycle_on({next_agent})"
    return True, "ok"


def run_safe(start: str, state: dict, breaker: dict) -> str:
    current = start
    while True:
        next_agent = AGENTS[current](state)
        ok, reason = breaker_check(breaker, next_agent)
        if not ok:
            state["log"].append(f"BREAKER FIRED: {reason}")
            return f"safe_exit:{reason}"
        current = next_agent
        if state.get("force_done"):
            return "normal_exit"


if __name__ == "__main__":
    print("=== UNSAFE run (no breaker) ===")
    state = {"log": []}
    t0 = time.perf_counter()
    result = run_unsafe("router", state, hard_iter_cap=50)
    print(f"  result={result}, hops={len(state['log'])}, elapsed={time.perf_counter()-t0:.2f}s")
    print(f"  last 4 hops: {state['log'][-4:]}")

    print("\n=== SAFE run (hop limit + cycle + watchdog) ===")
    state = {"log": []}
    breaker = make_breaker(max_hops=20, cycle_window=4, wallclock_seconds=2.0)
    t0 = time.perf_counter()
    result = run_safe("router", state, breaker)
    print(f"  result={result}, hops={breaker['hops']}, elapsed={time.perf_counter()-t0:.2f}s")
    print(f"  history: {list(breaker['history'])}")
    print(f"  last 4 events: {state['log'][-4:]}")

