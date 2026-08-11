from openai import OpenAI
import json

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

print("=" * 60)
print("OLD WAY : Chat Completions — stateless, manual history")
print("=" * 60)

history = []

history.append({"role": "user", "content": "My name is Sachin."})
r1 = client.chat.completions.create(model="llama3.1:latest", messages=history)
reply1 = r1.choices[0].message.content
history.append({"role": "assistant", "content": reply1})
print("Turn 1:", reply1)

history.append({"role": "user", "content": "What is my name?"})
r2 = client.chat.completions.create(model="llama3.1:latest", messages=history)
reply2 = r2.choices[0].message.content
history.append({"role": "assistant", "content": reply2})
print("Turn 2:", reply2)

print("\n" + "=" * 60)
print("NEW WAY : Responses API — server-side state via previous_response_id")
print("=" * 60)

r3 = client.responses.create(
    model="llama3.1:latest",
    input=[{"role": "user", "content": "My name is Sachin."}]
)
print("Turn 1:", r3.output_text)

r4 = client.responses.create(
    model="llama3.1:latest",
    previous_response_id=r3.id,
    input=[{"role": "user", "content": "What is my name?"}]
)
print("Turn 2:", r4.output_text)

print("\n" + "=" * 60)
print("RESPONSES API : built-in tool execution, no manual dispatch loop")
print("=" * 60)

weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
    }
}

def get_weather(city: str) -> str:
    return f"{city}: 22°C, sunny"

r5 = client.responses.create(
    model="llama3.1:latest",
    tools=[weather_tool],
    input=[{"role": "user", "content": "What is the weather in Pune?"}]
)

tool_outputs = []
for item in r5.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)
        result = get_weather(args["city"])
        tool_outputs.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": result
        })

r6 = client.responses.create(
    model="llama3.1:latest",
    previous_response_id=r5.id,
    tools=[weather_tool],
    input=tool_outputs
)
print("Answer:", r6.output_text)
