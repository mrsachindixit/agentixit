from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

puzzle = (
    "A farmer has 17 sheep. All but 9 die. How many sheep are left? "
    "Also: if a doctor gives you 3 pills and tells you to take one every half hour, "
    "how long before all pills are taken?"
)

print("=" * 60)
print("STANDARD MODEL : llama3.1")
print("=" * 60)

standard_response = client.chat.completions.create(
    model="llama3.1:latest",
    messages=[{"role": "user", "content": puzzle}]
)
print(standard_response.choices[0].message.content)

print("\n" + "=" * 60)
print("THINKING MODEL : lfm2.5-thinking (reasoning exposed)")
print("=" * 60)

thinking_response = client.chat.completions.create(
    model="lfm2.5-thinking:latest",
    messages=[{"role": "user", "content": puzzle}]
)

full = thinking_response.choices[0].message.content

if "<think>" in full and "</think>" in full:
    think_start = full.index("<think>") + len("<think>")
    think_end = full.index("</think>")
    reasoning = full[think_start:think_end].strip()
    answer = full[think_end + len("</think>"):].strip()
    print("[ REASONING TRACE ]")
    print(reasoning)
    print("\n[ FINAL ANSWER ]")
    print(answer)
else:
    print(full)
