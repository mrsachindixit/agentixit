# factor -> (principle, the smell when it's missing). Edit/extend and re-run.
FACTORS = {
    "1. Clear goal": (
        "The agent has one explicit, testable objective per run.",
        "Vague prompts; you can't tell if a run 'succeeded'.",
    ),
    "2. Prompt control": (
        "System/user prompts are versioned and owned, not ad-hoc strings.",
        "Prompts edited inline everywhere; no way to roll back a regression.",
    ),
    "3. Tool contracts": (
        "Tools have typed inputs/outputs and documented failure modes.",
        "Tools return free-form text the model has to guess at.",
    ),
    "4. State management": (
        "Conversation/run state is explicit and reconstructable.",
        "State lives only in memory; a restart loses everything.",
    ),
    "5. Memory strategy": (
        "You decide what to remember, summarize, and forget.",
        "Context window grows until it overflows or costs explode.",
    ),
    "6. Safety guardrails": (
        "Inputs and outputs are checked for injection, PII, and harm.",
        "Anything the model says goes straight to the user.",
    ),
    "7. Least privilege": (
        "Each tool/credential has the minimum scope it needs.",
        "One leaked prompt can trigger a wire transfer.",
    ),
    "8. Observability": (
        "Every call logs latency, tokens, tool use, and outcome.",
        "When it breaks in prod you have no idea why.",
    ),
    "9. Evaluation loop": (
        "A golden set + metrics gate every prompt/model change.",
        "You ship prompt edits and 'hope' nothing regressed.",
    ),
    "10. Latency & cost control": (
        "Budgets, caching, routing, and timeouts are explicit.",
        "Costs and p95 latency are discovered on the invoice.",
    ),
    "11. Protocol interop": (
        "Agents/tools speak stable protocols (MCP, A2A) not glue code.",
        "Every integration is a bespoke, brittle adapter.",
    ),
    "12. Versioning & governance": (
        "Prompts, models, and tools are versioned and auditable.",
        "Nobody can answer 'what changed?' after an incident.",
    ),
}


if __name__ == "__main__":
    print("THE 12 FACTORS OF A PRODUCTION AGENT\n")
    for name, (principle, smell) in FACTORS.items():
        print(name)
        print(f"   principle : {principle}")
        print(f"   smell      : {smell}\n")

    print("Self-check: for your current agent, which factors can you demo right now?")
    print("Every 'no' is a backlog item before you call it production-ready.")
