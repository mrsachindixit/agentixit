
### Using langchain agents with multiple tools and tool input schema validation using pydantic and decide tool call based on user input

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
import requests

model = ChatOllama(
    model="llama3.1:latest",  # Note: or "lfm2.5-thinking:latest" for better reasoning
    base_url='http://localhost:11434'
)

@tool
def get_pincode(city):
    """Simple HTTP call to a public pincode service
    
    Args:
        city (str): The city to get the pincode for.
        
    Returns:
        str: The pincode for the given city.
    """
    print(f"— Tool Call Triggered for: {city}")
    result = f"{city}: 123456"
    print(f"— Real API Result: {result}")
    return result

@tool
def get_weather(city):
    """Simple HTTP call to a public weather service
    
    Args:
        city (str): The city to get the weather for.
        
    Returns:
        str: The weather for the given city.
    """
    print(f"— Tool Call Triggered for: {city}")
    result = f"{city}: +12°C"
    print(f"— Real API Result: {result}")
    return result

agent = create_agent(
    model=model,
    tools=[get_pincode, get_weather],
    system_prompt = "Try to choose appropriate tool to call. Summarize the tool response for user.",
    # debug=True
)

def run_agent(user_message):
    messages = [
        {"role": "user", "content": user_message}
    ]
    print("— Asking Ollama...")

    ai_msg = agent.invoke({"messages": messages})

    return ai_msg.get('messages')[-1].content

if __name__ == "__main__":
    query = "What is the pincode for Bogotá, Colombia?"
    print(f"User: {query}")
    answer1 = run_agent(query)
    print(f"Agent: {answer1}")
    print("="*60)
    query = "What's the weather in Bogotá, Colombia in celsius?"
    print(f"User: {query}")
    answer2 = run_agent(query)
    print(f"Agent: {answer2}")

