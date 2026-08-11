 
import re
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from typing import Any, Optional
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage

# 1. Configure the model
# Use a capable model like llama3 or "lfm2.5-thinking:latest" for better reasoning
model = ChatOllama(
    model="llama3.1:latest",
    base_url='http://localhost:11434'
)
class SecretMaskerMiddleware(AgentMiddleware):
    """Middleware to redact sensitive patterns from user input."""
    
    def before_model(self, state: AgentState, runtime: Any) -> Optional[dict[str, Any]]:
        # Check if the last message is from a user
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage):
            # Regex to find anything starting with 'SECRET_'
            sanitized_content = re.sub(r'SECRET_\w+', '[REDACTED]', last_msg.content)
            
            if sanitized_content != last_msg.content:
                print(f"--- [Middleware] Redacted sensitive info! ---")
                # Update the state with the sanitized message
                state["messages"][-1].content = sanitized_content
                
        return None  # Continue execution with updated state

# --- Setup and Test ---
agent = create_agent(
    model=model, 
    tools=[], 
    middleware=[SecretMaskerMiddleware()]
)

# Test with a sensitive 'key' in the string
user_input = "My password is SECRET_12345, please remember it."
response = agent.invoke({"messages": [HumanMessage(content=user_input)]})

# The model only saw: "My password is [REDACTED], please remember it."
print(f"Agent Response: {response['messages'][-1].content}")