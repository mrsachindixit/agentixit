from typing import TypedDict, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


class CoordinatorState(TypedDict, total=False):
    request: str
    decision: str
    output: str


def booking_handler(request: str) -> str:
    print("\n--- DELEGATING TO BOOKING HANDLER ---")
    return (
        f"Booking Handler processed request: '{request}'. "
        "Result: Simulated booking action."
    )


def info_handler(request: str) -> str:
    print("\n--- DELEGATING TO INFO HANDLER ---")
    return (
        f"Info Handler processed request: '{request}'. "
        "Result: Simulated information retrieval."
    )


def unclear_handler(request: str) -> str:
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify."


def build_graph(llm: ChatOllama):
    router_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Analyze the user's request and determine which
specialist handler should process it.
- If the request is related to booking flights or hotels,
output 'booker'.
- For all other general information questions, output 'info'.
- If the request is unclear or doesn't fit either category,
output 'unclear'.
ONLY output one word: 'booker', 'info', or 'unclear'.""",
            ),
            ("user", "{request}"),
        ]
    )

    def route_node(state: CoordinatorState) -> CoordinatorState:
        msg = router_prompt.format_messages(request=state["request"])
        response = llm.invoke(msg)
        decision = response.content.strip().lower() if response and response.content else "unclear"
        if decision not in {"booker", "info", "unclear"}:
            decision = "unclear"
        return {"decision": decision}

    def booker_node(state: CoordinatorState) -> CoordinatorState:
        return {"output": booking_handler(state["request"])}

    def info_node(state: CoordinatorState) -> CoordinatorState:
        return {"output": info_handler(state["request"])}

    def unclear_node(state: CoordinatorState) -> CoordinatorState:
        return {"output": unclear_handler(state["request"])}

    def route_selector(state: CoordinatorState) -> Literal["booker", "info", "unclear"]:
        decision = state.get("decision", "unclear").strip().lower()
        if decision == "booker":
            return "booker"
        if decision == "info":
            return "info"
        return "unclear"

    graph = StateGraph(CoordinatorState)
    graph.add_node("router", route_node)
    graph.add_node("booker", booker_node)
    graph.add_node("info", info_node)
    graph.add_node("unclear", unclear_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_selector)
    graph.add_edge("booker", END)
    graph.add_edge("info", END)
    graph.add_edge("unclear", END)

    return graph.compile()
# START -> router -> {booker, info, unclear} -> END

def main():
    try:
        llm = ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434")
        print(f"Language model initialized: {llm.model}")
    except Exception as e:
        print(f"Error initializing language model: {e}")
        print("\nSkipping execution due to LLM initialization failure.")
        return

    coordinator_graph = build_graph(llm)

    print("--- Running with a booking request ---")
    request_a = "Book me a flight to London."
    result_a = coordinator_graph.invoke({"request": request_a})
    print(f"Final Result A: {result_a['output']}")

    print("\n--- Running with an info request ---")
    request_b = "What is the capital of Italy?"
    result_b = coordinator_graph.invoke({"request": request_b})
    print(f"Final Result B: {result_b['output']}")

    print("\n--- Running with an unclear request ---")
    request_c = "Tell me about quantum physics."
    result_c = coordinator_graph.invoke({"request": request_c})
    print(f"Final Result C: {result_c['output']}")


if __name__ == "__main__":
    main()
