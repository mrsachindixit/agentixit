

import requests




# PROMPT STYLES - Try these to see how prompts shape outputs:

# Simple question
# prompt = "Explain the concept of AI agents in one sentence."

# Structured request
# prompt = "Summarize the benefits of unit tests in 3 bullet points."

# Translation task
# prompt = "Translate to Spanish: 'Where is the train station?'"

# Classification task
# prompt = "Classify the sentiment (positive/neutral/negative): 'The service was slow but the food was great.'"

# Creative generation
# prompt = "Write a short (3-line) poem about rain."

# Information extraction
# prompt = "Extract keywords from this text: 'Agents combine reasoning and tools to solve tasks.'"

# JSON output (format constraint)
# prompt = "Given input JSON {\"a\": 3, \"b\": 4}, output only the sum as a number."

# Strict JSON generation
# prompt = "You are a strict JSON generator. Output only JSON with keys: title, summary. Topic: LangChain."

url = "http://localhost:11434/api/generate"

# Prompt examples: uncomment ONE at a time to try different prompt styles.
prompt = "What is the capital of France?"

payload = {
    "model": "llama3.1:latest",  # Default model
    "prompt": prompt,
    "stream": False
}
print(f"User: {prompt}")
response = requests.post(url, json=payload)

print(f"Assistant: {response.json().get('response', '')}")


# ---------------- LLM FLAGS  pls tweak them and read about them ----------------
 

# Example 1: Add options in payload
# payload_with_flags = {
#     "model": "llama3.1:latest",
#     "prompt": "Write one short motivational line.",
#     "stream": False,
#     "options": {
#         "temperature": 0.7,
#         "top_p": 0.9,
#         "top_k": 40,
#         "num_predict": 60,
#         "repeat_penalty": 1.1,
#         "seed": 42
#     }
# }
# response = requests.post(url, json=payload_with_flags)
# print(response.json().get("response", ""))

# Example 2: Change one flag at a time
# payload_top_p = {
#     "model": "llama3.1:latest",
#     "prompt": "Write one short motivational line.",
#     "stream": False,
#     "options": {"top_p": 0.6}
# }
# response = requests.post(url, json=payload_top_p)
# print(response.json().get("response", ""))


