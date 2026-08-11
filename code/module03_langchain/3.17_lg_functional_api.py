from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

model = ChatOllama(model="llama3.1:latest", base_url='http://localhost:11434')

print("=" * 60)
print("STATEGRAPH WAY (from 3.5 / 3.10)")
print("=" * 60)

class State(MessagesState):
    pass

agent_sg = create_agent(model=model, tools=[])

def agent_node(state: State):
    result = agent_sg.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

graph = (
    StateGraph(State)
    .add_node("agent", agent_node)
    .add_edge(START, "agent")
    .add_edge("agent", END)
    .compile(checkpointer=InMemorySaver())
)

config_sg = {"configurable": {"thread_id": "sg-thread-1"}}
out_sg = graph.invoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]}, config_sg)
print("StateGraph answer:", out_sg["messages"][-1].content)

print("\n" + "=" * 60)
print("FUNCTIONAL API WAY  (@task + @entrypoint)")
print("=" * 60)

checkpointer = InMemorySaver()

@task
def call_model(user_message: str) -> str:
    agent_fn = create_agent(model=model, tools=[])
    result = agent_fn.invoke({"messages": [{"role": "user", "content": user_message}]})
    return result["messages"][-1].content

@entrypoint(checkpointer=checkpointer)
def pipeline(user_message: str) -> str:
    answer = call_model(user_message).result()
    return answer

config_fn = {"configurable": {"thread_id": "fn-thread-1"}}
out_fn = pipeline.invoke("What is 2 + 2?", config_fn)
print("Functional answer:", out_fn)

print("\n" + "=" * 60)
print("FUNCTIONAL API : multi-task parallel execution")
print("=" * 60)

@task
def translate_fr(text: str) -> str:
    agent_t = create_agent(model=model, tools=[])
    r = agent_t.invoke({"messages": [{"role": "user", "content": f"Translate to French: {text}"}]})
    return r["messages"][-1].content

@task
def translate_de(text: str) -> str:
    agent_t = create_agent(model=model, tools=[])
    r = agent_t.invoke({"messages": [{"role": "user", "content": f"Translate to German: {text}"}]})
    return r["messages"][-1].content

@entrypoint(checkpointer=InMemorySaver())
def multi_translate(text: str) -> dict:
    fr_future = translate_fr(text)
    de_future = translate_de(text)
    return {"french": fr_future.result(), "german": de_future.result()}

config_mt = {"configurable": {"thread_id": "mt-thread-1"}}
translations = multi_translate.invoke("Hello, how are you?", config_mt)
print("French :", translations["french"])
print("German :", translations["german"])
