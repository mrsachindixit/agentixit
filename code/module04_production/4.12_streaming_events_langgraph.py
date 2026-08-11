import time
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


llm = ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434", temperature=0)


def plan_node(state: dict) -> dict:
    plan = llm.invoke(f"Outline 3 bullet steps to answer: {state['question']}").content
    return {"plan": plan}


def answer_node(state: dict) -> dict:
    answer = llm.invoke(
        f"Following this plan, answer the question concisely.\n"
        f"Plan:\n{state['plan']}\n\nQuestion: {state['question']}"
    ).content
    return {"answer": answer}


graph = (
    StateGraph(dict)
    .add_node("plan",   plan_node)
    .add_node("answer", answer_node)
    .add_edge(START, "plan")
    .add_edge("plan",  "answer")
    .add_edge("answer", END)
    .compile()
)


def demo_state_events(question: str) -> None:
    print("\n=== mode='updates' (state-level) ===")
    for event in graph.stream({"question": question}, stream_mode="updates"):
        for node, delta in event.items():
            preview = ""
            if delta:
                first_value = list(delta.values())[0]   # the one field this node wrote
                preview = str(first_value)[:70]
            print(f"  [event] node={node!r:<12} preview={preview!r}")


def demo_token_stream(question: str) -> None:
    print("\n=== mode='messages' (token-level) ===")
    started = time.perf_counter()
    first_token_ms = None
    for chunk, meta in graph.stream({"question": question}, stream_mode="messages"):
        if hasattr(chunk, "content") and chunk.content:
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            print(chunk.content, end="", flush=True)
    print(f"\n  [first-token-latency ~ {first_token_ms:.0f} ms]")


def demo_cancellation(question: str, kill_after_ms: int = 800) -> None:
    print("\n=== cancellation ===")
    deadline = time.perf_counter() + kill_after_ms / 1000

    cancelled = False
    for chunk, _ in graph.stream({"question": question}, stream_mode="messages"):
        if time.perf_counter() > deadline:
            cancelled = True
            print("\n  [CANCELLED by deadline]")
            break
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)
    if not cancelled:
        print("\n  [completed before deadline]")


if __name__ == "__main__":
    q = "Explain why prompt caching reduces cost for long system prompts."

    demo_state_events(q)
    demo_token_stream(q)
    demo_cancellation(q, kill_after_ms=1200)
