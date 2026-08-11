import time

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_unfocused_messages(user_query: str):
    system = SystemMessage(
        content=(
            "You are a helpful assistant. Be very detailed. Include caveats, history, alternatives, "
            "a glossary, and extended background unless user asks otherwise."
        )
    )

    history = [
        HumanMessage(content="Hi, can you help me later with architecture?"),
        AIMessage(content="Sure. Happy to help with architecture and planning."),
        HumanMessage(content="Also compare many cloud providers in depth."),
        AIMessage(content="I can compare all major providers with full detail."),
        HumanMessage(content="Remember my team likes concise updates but full technical depth in final docs."),
        AIMessage(content="Understood. I will keep that preference in mind for future responses."),
        HumanMessage(content="I might ask something about retries, caching, and async execution."),
        AIMessage(content="Great topics. I can explain all with examples and tradeoffs."),
    ]

    current = HumanMessage(content=user_query)
    return [system, *history, current]


def build_focused_messages(user_query: str):
    system = SystemMessage(
        content=(
            "You are a production AI assistant. "
            "Return exactly 5 bullets, each under 14 words. "
            "No intro, no conclusion, no markdown table."
        )
    )

    compact_memory = HumanMessage(
        content=(
            "Context summary: user cares about production tradeoffs, wants concise actionable output."
        )
    )

    current = HumanMessage(content=user_query)
    return [system, compact_memory, current]


def run_once(llm: ChatOllama, label: str, messages):
    serialized = "\n".join(m.content for m in messages)
    token_estimate = approx_tokens(serialized)

    started = time.perf_counter()
    response = llm.invoke(messages)
    elapsed = time.perf_counter() - started

    text = response.content if isinstance(response.content, str) else str(response.content)
    preview = text.strip().replace("\n", " ")[:160]

    return {
        "label": label,
        "elapsed": elapsed,
        "approx_input_tokens": token_estimate,
        "answer_preview": preview,
    }


def print_report(before, after):
    print("=" * 90)
    print("Prompt/Role/Memory Performance Demo")
    print("=" * 90)
    print(f"{'Scenario':<18} {'Elapsed(s)':<12} {'ApproxInputTokens':<18}")
    print("-" * 90)
    print(f"{before['label']:<18} {before['elapsed']:<12.2f} {before['approx_input_tokens']:<18}")
    print(f"{after['label']:<18} {after['elapsed']:<12.2f} {after['approx_input_tokens']:<18}")

    speedup = before["elapsed"] / max(after["elapsed"], 0.001)
    token_reduction = 100.0 * (before["approx_input_tokens"] - after["approx_input_tokens"]) / max(before["approx_input_tokens"], 1)

    print("-" * 90)
    print(f"Speedup (focused / unfocused): {speedup:.2f}x")
    print(f"Approx input-token reduction:   {token_reduction:.1f}%")

    print("\nAnswer preview (unfocused):")
    print(before["answer_preview"])
    print("\nAnswer preview (focused):")
    print(after["answer_preview"])


if __name__ == "__main__":
    llm = ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434", temperature=0)

    query = "Give practical performance optimization steps for an AI support assistant."

    unfocused_messages = build_unfocused_messages(query)
    focused_messages = build_focused_messages(query)

    run_unfocused = run_once(llm, "Unfocused", unfocused_messages)
    run_focused = run_once(llm, "Focused", focused_messages)

    print_report(run_unfocused, run_focused)
