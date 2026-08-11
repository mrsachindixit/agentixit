
import os
# Set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "your_api_key"

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. Configure the model
# Use a capable model like GPT-4o-mini for efficient agentic reasoning
model = ChatOllama(model="llama3.1:latest",base_url='http://localhost:11434')

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

# 6. Invoke the agent to start a conversation
def run_agent(user_message, config=config):
    messages = [
        {"role": "user", "content": user_message}
    ]
    print("— Asking Ollama...")

    ai_msg = agent.invoke({"messages": messages}, config)

    return ai_msg.get('messages')[-1].content


# First message
query = "Hi, my name is Sumit. I am an architect."
print(f"--- User : {query} ---")
response1 = run_agent(query)
# The agent will not know your name in this first turn, as it hasn't saved it yet.
print("Agent :", response1)


# Second message (in the same thread)
query2 = "Based on our previous conversation, what is my name?"
print(f"\n--- User : {query2} ---")
response2 = run_agent(query2)
# Now the agent can access the previous turn and the provided name
print("Agent :", response2)


# Third message (new thread)
query3 = "Hello! What is my name?"
print(f"\n--- User  (New Session): {query3} ---")
# Use a different thread_id for a new, separate conversation
new_user_session = {"configurable": {"thread_id": "user_session_2"}}
response3 = run_agent(query3, config=new_user_session)
# In this new session, the agent has no prior context about "Bob"
print("Agent 3:", response3)
