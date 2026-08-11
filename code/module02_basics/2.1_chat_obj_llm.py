

from openai import OpenAI

# Initialize client pointing to local Ollama with OpenAI-compatible endpoint
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Required for OpenAI SDK, but can be any string for local Ollama
)
prompt = "In one paragraph, explain what an AI agent is to a new developer."

print(f"User : {prompt}")

# Send a simple chat request to the LLM
response = client.chat.completions.create(
    model="llama3.1:latest",  # Default model
    messages=[{
        "role": "user",
        "content": prompt
    }]
)


# Extract and print the response
print(f"Assistant: {response.choices[0].message.content}")




