
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from typing import Dict

model = ChatOllama(
    model="llama3.1:latest",  # Note: or "lfm2.5-thinking:latest" for better reasoning
    base_url='http://localhost:11434'
)

agent = create_agent(model=model, tools=[])

class State(dict):  
    query: str

def agent_node(state: State) -> dict:
    """A LangGraph node that invokes a LangChain agent."""
    print(f"Invoking agent with query: {state['query']}")
    query = state.get("query", "")
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    output = {"answer": result["messages"][-1].content}
    print(f"Agent returned answer: {output['answer']}")
    return output

# Build a simple workflow
workflow = (
    StateGraph(State)
    .add_node("agent", agent_node)
    .add_edge(START, "agent")
    .add_edge("agent", END)
    .compile()
)
# START -> agent -> END
result = workflow.invoke({"query": "What is the capital of France?"})
print(f"Agent's answer: {result}")