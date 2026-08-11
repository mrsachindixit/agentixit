from langchain.tools import tool
from langchain.agents import create_agent
from langchain_deepagents import DeepAgent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOllama(model="llama3.1:latest", base_url='http://localhost:11434')

@tool
def get_user_details(user_id: str) -> dict:
    """Get user details for a given user ID."""
    if user_id == "user123":
        return {"user_id": user_id, "astrological_sign": "Aquarius", "country": "USA"}
    return {"user_id": user_id, "astrological_sign": "Leo", "country": "UK"}

@tool
def get_membership(user_id: str) -> dict:
    """Get membership details for a given user ID."""
    if user_id == "user123":
        return {"user_id": user_id, "membership_type": "Basic"}
    return {"user_id": user_id, "membership_type": "Premium"}

@tool
def get_horoscope(sign: str) -> str:
    """Fetch today's horoscope for a given astrological sign."""
    return f"{sign}: Today is a great day for new beginnings."

tools = [get_user_details, get_membership, get_horoscope]

print("=" * 60)
print("STANDARD AGENT (create_agent) — manual context, no compression")
print("=" * 60)

standard_agent = create_agent(model=model, tools=tools, checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "std-1"}}

for msg in [
    "Get details for user123.",
    "What is their membership?",
    "What is their horoscope today?",
    "Summarise everything you know about user123.",
]:
    result = standard_agent.invoke({"messages": [{"role": "user", "content": msg}]}, config)
    print(f"Q: {msg}")
    print(f"A: {result['messages'][-1].content}\n")

print("=" * 60)
print("DEEP AGENT — auto context compression, virtual filesystem, subagent spawning")
print("=" * 60)

deep_agent = DeepAgent(
    model=model,
    tools=tools,
)

deep_result = deep_agent.invoke(
    "Get details, membership and horoscope for user123, then write a short profile report."
)
print("Deep Agent result:")
print(deep_result)

print("\n" + "=" * 60)
print("DEEP AGENT : filesystem scratch — agent writes intermediate results")
print("=" * 60)

deep_agent_fs = DeepAgent(
    model=model,
    tools=tools,
    enable_filesystem=True,
)

fs_result = deep_agent_fs.invoke(
    "Fetch details and membership for user456. "
    "Save a summary to a file called user456_summary.txt, then read it back and print the contents."
)
print(fs_result)
