from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:latest",  # Note: or "lfm2.5-thinking:latest" for better reasoning
    base_url='http://localhost:11434'
)

agent = create_agent(model=model, tools=[])


class State(dict):
    query: str
    answer: str
    approved: bool


def agent_node(state: State) -> dict:
    """Get answer from agent."""
    print(f"Invoking agent with query: {state['query']}")
    query = state.get("query", "")
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    output = {"answer": result["messages"][-1].content}
    print(f"Agent returned answer: {output['answer']}")
    return output


def human_gate(state: State) -> dict:
    """Pause and ask for human approval."""
    decision = interrupt({
        "question": "Approve this answer? (y/n)",
        "answer": state.get("answer", "")
    })
    approved = str(decision).strip().lower().startswith("y")
    return {"approved": approved}


workflow = (
    StateGraph(State)
    .add_node("agent", agent_node)
    .add_node("human_gate", human_gate)
    .add_edge(START, "agent")
    .add_edge("agent", "human_gate")
    .add_edge("human_gate", END)
    .compile(checkpointer=MemorySaver())
)

config = {"configurable": {"thread_id": "lesson-3-16"}}

paused_state = workflow.invoke({"query": "What is the capital of France?"}, config=config)
print("\nGraph paused for human review.")
print(f"Current state: {paused_state}")

human_input = input("Approve answer? Type y or n: ").strip()
final_state = workflow.invoke(Command(resume=human_input), config=config)

print(f"\nFinal state: {final_state}")
