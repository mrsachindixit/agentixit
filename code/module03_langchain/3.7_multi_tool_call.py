### Multiple tools decide the tool based on user input

from langchain.tools import tool
from langchain_ollama import ChatOllama
import requests
from pydantic import BaseModel, Field

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
    return f"{city}: 123456"

@tool
def get_weather(city):
    """Simple HTTP call to a public weather service
    
    Args:
        city (str): The city to get the weather for.
        
    Returns:
        str: The weather for the given city.
    """
    return f"{city}: +12°C"

model_with_tools = model.bind_tools([get_pincode, get_weather])

# messages = [
#     {"role": "user", "content": "What is my horoscope today? I am an Aquarius."}
# ]
def run_agent(user_message):
    messages = [
        {"role": "user", "content": user_message}
    ]

    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)
    print("— Asking Ollama...")
    # Step 2: Execute tools and collect results
    # Execute the tool with the generated arguments
    for tool_call in  ai_msg.tool_calls:
        if tool_call.get("name") == "get_pincode":
            print(f"— Tool Call Triggered for: {tool_call['args']}")
            tool_result = get_pincode.invoke(tool_call)
            print(f"— Real API Result: {tool_result.content}")
            messages.append(tool_result)
        if tool_call.get("name") == "get_weather":
            print(f"— Tool Call Triggered for: {tool_call['args']}")
            tool_result = get_weather.invoke(tool_call)
            print(f"— Real API Result: {tool_result.content}")
            messages.append(tool_result)

    print("\n— Sending results back to Ollama for final answer...")
    # Step 3: Pass results back to model for final response
    ai_msg = model_with_tools.invoke(messages)
    return ai_msg.text

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