

import requests


# 1. Define the tool configuration (The Schema)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"}
            },
            "required": ["city"]
        }
    }
}]

def get_weather(city):
    """Simulated weather API call."""
    return f"{city}: +12°C"

def run_agent(user_message):
    """
    Run the agent loop:
    Query → LLM decides → Tool execution → Final answer
    """
    # 2. Ask Ollama a question WITH tool definitions
    messages = [
        {
            "role": "system",
            "content": "You are a helpful, step-by-step assistant. You can consider data returned from tool is latest."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    print(f"— Asking about user query to assistant...")
    res = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.1:latest",
        "messages": messages,
        "tools": tools,
        "stream": False
    }).json()

    # 3. Check if the model wants to use the tool
    if "tool_calls" in res["message"]:
        tool_call = res["message"]["tool_calls"][0]
        city = tool_call["function"]["arguments"]["city"]
        
        print(f"— Tool Call Triggered for getting weather of {city}")
        
        # Execute the actual weather fetch
        weather_info = get_weather(city)
        print(f"— The weather info received : {weather_info}")

        # 4. Send the result back to Ollama for the final answer
        messages.append(res["message"])  # Add the model's request
        messages.append({"role": "tool", "content": weather_info})  # Add our result
        
        print("\n— Processing the response...")
        final_res = requests.post("http://localhost:11434/api/chat", json={
            "model": "llama3.1:latest",
            "messages": messages,
            "stream": False
        }).json()

        return final_res["message"]["content"]
    else:
        # Model didn't need tool, return direct answer
        return res["message"]["content"]

if __name__ == "__main__":
    user_question = "What's the weather in Bogotá, Colombia in celsius?"
    print(f"User : {user_question}\n")
    answer = run_agent(user_question)
    print(f"Final Answer : {answer}")
