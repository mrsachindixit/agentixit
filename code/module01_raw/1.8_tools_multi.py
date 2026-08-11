

import requests

def get_pincode(city):
    """Simulated pincode API call."""
    return f"{city}: 123456"

def get_weather(city):
    """Simulated weather API call."""
    return f"{city}: +12°C"


# Define multiple tools as array
tools = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_pincode",
            "description": "Get the pincode for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"}
                },
                "required": ["city"]
            }
        }
    }
]


def run_agent(user_message):
    """
    Multi-tool agent loop:
    1. Send query with all tool defs
    2. LLM selects which tool(s) needed
    3. Execute selected tools
    4. Send results back for final answer
    """
    messages = [
        {"role": "system", "content": "You are a helpful, step-by-step assistant. You can consider data returned from tool is latest."},
        {"role": "user", "content": user_message}
    ]
    
    print(f"— Asking about user query to assistant...")
    res = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.1:latest",
        "messages": messages,
        "tools": tools,
        "stream": False
    }).json()

    # Handle Tool Calls (can be multiple)
    messages.append(res["message"])
    
    if "tool_calls" in res["message"]:
        for tool_call in res["message"]["tool_calls"]:
            function_name = tool_call["function"]["name"]
            city = tool_call["function"]["arguments"]["city"]
            
            
            
            # Execute appropriate tool
            if function_name == "get_weather":
                print(f"— Tool Call Triggered for getting weather of {city}")
                result = get_weather(city)
            elif function_name == "get_pincode":
                print(f"— Tool Call Triggered for getting pincode of {city}")
                result = get_pincode(city)
            else:
                result = "Unknown tool"
            
            messages.append({"role": "tool", "tool_name": function_name, "content": result})
        print(f"— Tool response received. Assistant now processing the response for final answer...")
        
    final_res = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.1:latest",
        "messages": messages,
        "stream": False
    }).json()

    return final_res['message']['content']

if __name__ == "__main__":
    user_question = "What is the weather and pincode of Berlin?"
    print(f"User: {user_question}\n")
    answer = run_agent(user_question)
    print(f"Agent: {answer}")
    
    print("\n" + "="*60 + "\n")
    
    user_question = "Tell me the weather in London."
    print(f"User: {user_question}\n")
    answer = run_agent(user_question)
    print(f"Agent: {answer}")
    