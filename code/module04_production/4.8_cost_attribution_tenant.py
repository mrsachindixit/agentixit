import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import chat


PRICES = {
    "ollama/llama3.1:latest":   {"in": 0.0,   "out": 0.0},   # local -> 0 cost (still log tokens!)
    "openai/gpt-4o-mini":       {"in": 0.15,  "out": 0.60},
    "openai/gpt-4o":            {"in": 2.50,  "out": 10.00},
    "anthropic/claude-sonnet-4-5": {"in": 3.00, "out": 15.00},
}

# Every tracked call appends one row here; chargeback() later sums these rows by tag.
LEDGER = []


def estimate_tokens(text):
    return max(1, len(text) // 4)


# Pass who the call is for (tags) explicitly, so its cost can be attributed to them.
def tracked_chat(messages, tags, model_key="ollama/llama3.1:latest"):
    started = time.perf_counter()
    answer = chat(messages, model=model_key.split("/", 1)[1])
    elapsed_ms = (time.perf_counter() - started) * 1000

    in_tokens = sum(estimate_tokens(m["content"]) for m in messages)
    out_tokens = estimate_tokens(answer)
    price = PRICES[model_key]
    cost = (in_tokens * price["in"] + out_tokens * price["out"]) / 1_000_000

    row = {
        "model": model_key,
        "in_tok": in_tokens,
        "out_tok": out_tokens,
        "cost_usd": cost,
        "ms": round(elapsed_ms, 1),
    }
    row.update(tags)            # add tenant/feature/etc. onto this row
    LEDGER.append(row)
    return answer


def chargeback(group_by):
    # Sum calls/tokens/cost for every distinct combination of the group_by tags.
    buckets = {}
    for row in LEDGER:
        key = tuple(row.get(g) for g in group_by)
        if key not in buckets:
            buckets[key] = {"calls": 0, "in_tok": 0, "out_tok": 0, "cost_usd": 0.0}
        bucket = buckets[key]
        bucket["calls"] += 1
        bucket["in_tok"] += row["in_tok"]
        bucket["out_tok"] += row["out_tok"]
        bucket["cost_usd"] += row["cost_usd"]

    # Turn each bucket back into a flat row: the group_by values + the summed metrics.
    report = []
    for key, totals in sorted(buckets.items()):
        line = dict(zip(group_by, key))
        line["calls"] = totals["calls"]
        line["in_tok"] = totals["in_tok"]
        line["out_tok"] = totals["out_tok"]
        line["cost_usd"] = round(totals["cost_usd"], 4)
        report.append(line)
    return report


if __name__ == "__main__":
    for tenant, feature, n in [("acme", "chat", 2), ("acme", "summarize", 1), ("globex", "chat", 3)]:
        for i in range(n):
            tracked_chat(
                [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": f"{feature} request {i} for {tenant}"},
                ],
                tags={"tenant": tenant, "feature": feature},
            )

    print("=== Per-tenant chargeback ===")
    for row in chargeback(["tenant"]):
        print(row)

    print("\n=== Per-tenant per-feature ===")
    for row in chargeback(["tenant", "feature"]):
        print(row)

    print("\n=== Per-model (helps capacity planning) ===")
    for row in chargeback(["model"]):
        print(row)

