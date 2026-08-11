import hashlib
import threading
import time

from utils.ollama_client import chat


# Shared state: which prompts are currently being answered, and their results.
_inflight: dict[str, threading.Event] = {}
_results: dict[str, str] = {}
_lock = threading.Lock()


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def call_model(prompt: str) -> str:
    return chat([{"role": "user", "content": prompt}]).strip()


def ask_naive(prompt: str) -> str:
    # Every caller does its own model call - duplicate work.
    return call_model(prompt)


def ask_coalesced(prompt: str) -> str:
    key = _key(prompt)

    with _lock:
        if key in _inflight:
            event, am_owner = _inflight[key], False   # someone else is already fetching
        else:
            event, am_owner = threading.Event(), True  # I'm first -> I'll fetch
            _inflight[key] = event

    if am_owner:
        try:
            _results[key] = call_model(prompt)
        finally:
            event.set()             # wake up everyone waiting
            with _lock:
                _inflight.pop(key, None)
    else:
        event.wait(timeout=120)     # piggyback: wait for the owner's result

    return _results.get(key, "[no result]")


def race(fn, prompt: str, n: int = 5) -> float:
    # Fire n identical requests at the same time, return total wall-clock seconds.
    start = time.perf_counter()
    threads = [threading.Thread(target=fn, args=(prompt,)) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - start


if __name__ == "__main__":
    prompt = "In exactly 5 bullets, explain SLO monitoring for AI agents."

    naive_secs = race(ask_naive, prompt, n=5)
    print(f"Naive (5 separate model calls): {naive_secs:.2f}s")

    coalesced_secs = race(ask_coalesced, prompt, n=5)
    print(f"Coalesced (1 model call shared): {coalesced_secs:.2f}s")
