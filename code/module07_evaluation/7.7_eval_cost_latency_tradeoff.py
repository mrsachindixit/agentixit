import time

from utils.ollama_client import chat


def llm_answer(system_prompt, user_prompt, options=None):
    started = time.perf_counter()
    try:
        text = chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **(options or {}),
        ).strip()
    except Exception as exc:
        text = f"ERROR: {exc}"
    return text, (time.perf_counter() - started) * 1000


def score_accuracy(answer, required_terms):
    text = answer.lower()
    hits = sum(1 for term in required_terms if term.lower() in text)
    return hits / max(len(required_terms), 1)


# Two trade-offs, side by side: accuracy you buy per second, and per unit cost.
def accuracy_per_second(run):
    return run["accuracy"] / max(run["latency_ms"] / 1000.0, 1e-6)


def accuracy_per_cost(run):
    return run["accuracy"] / max(run["cost_proxy"], 1e-6)


if __name__ == "__main__":
    query = "In the multi-agent flow, explain when to call horoscope_tool vs weather_tool for a premium user."
    required = ["premium", "horoscope_tool", "weather_tool"]

    # (name, system_prompt, options) - try tweaking num_predict to feel the trade-off.
    variant_specs = [
        ("fast",     "Answer in one short sentence.",        {"num_predict": 40,  "temperature": 0.1}),
        ("balanced", "Answer in two concise bullet points.", {"num_predict": 90,  "temperature": 0.2}),
        ("detailed", "Answer in four detailed bullet points.", {"num_predict": 180, "temperature": 0.3}),
    ]

    runs = []
    for name, system_prompt, options in variant_specs:
        answer, latency_ms = llm_answer(system_prompt, query, options)
        runs.append({
            "name": name,
            "accuracy": score_accuracy(answer, required),
            "latency_ms": latency_ms,
            "cost_proxy": max(len(answer) / 1000, 1e-3),  # cheap token proxy
        })

    print("Per-variant metrics:")
    for run in runs:
        print({
            "variant": run["name"],
            "accuracy": round(run["accuracy"], 2),
            "latency_ms": round(run["latency_ms"], 1),
            "cost": round(run["cost_proxy"], 3),
            "acc_per_sec": round(accuracy_per_second(run), 3),   # latency trade-off
            "acc_per_cost": round(accuracy_per_cost(run), 3),    # cost trade-off
        })

    best_latency = max(runs, key=accuracy_per_second)
    best_cost = max(runs, key=accuracy_per_cost)
    best_raw = max(runs, key=lambda r: r["accuracy"])

    print("\nWho wins on each axis:")
    print(f"  most accurate (ignore budget): {best_raw['name']}")
    print(f"  best accuracy per second      : {best_latency['name']}  <- pick for latency-sensitive apps")
    print(f"  best accuracy per cost        : {best_cost['name']}  <- pick for cost-sensitive batch jobs")

    if best_latency["name"] != best_cost["name"]:
        print("\nNote: the latency winner and the cost winner differ - there is no free lunch.")
    else:
        print("\nNote: same winner on both axes here, but that is not guaranteed in general.")
