
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. Configure the model
# Use a capable model like GPT-4o-mini for efficient agentic reasoning
model = ChatOllama(model="llama3.1:latest", base_url='http://localhost:11434')

# 2. Define tools (optional, but agents are useful with them)
# For this simple example, we'll use an empty list
tools = []

# 3. Initialize the InMemorySaver (the simple memory saver)
# This checkpointer will save the state in a dictionary in memory
checkpointer = InMemorySaver()

# 4. Create the agent with the checkpointer
# The agent will automatically manage the "messages" key in its state
agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=checkpointer,
)

# 5. Define a configuration for the conversation thread
# A unique thread_id is necessary to associate memory with a specific conversation
config = {"configurable": {"thread_id": "user_session_1"}}

def run_agent(user_message):
    messages = [
        {"role": "user", "content": user_message}
    ]
    print("— Asking Ollama...")

    ai_msg = agent.invoke({"messages": messages}, config)

    return ai_msg.get('messages')[-1].content


if __name__ == "__main__":
    print(run_agent("My name is Sachin."))
    print(run_agent("I work as an architect."))
    print(run_agent("Give info about my city"))
    print(run_agent("I live in Pune."))
    print(run_agent("Give info about my city"))
