

from langchain_ollama import ChatOllama
researcher = ChatOllama(
    model="lfm2.5-thinking:latest",  # Note: or "lfm2.5-thinking:latest" for better reasoning
    base_url='http://localhost:11434'
)

critic = ChatOllama(
    model="lfm2.5-thinking:latest",  # Note: or "lfm2.5-thinking:latest" for better reasoning
    base_url='http://localhost:11434',
    # verbose = True
)

def ask(model, system, user):
    print(f"== {system} == {user} ==")
    messages = [{"role":"system","content":system},
                {"role":"user","content":user}]
    return model.invoke(messages).content

if __name__ == "__main__":
    print("== Review and critique pattern ==")
    topic = "Multiply two digit numbers."
    print("="*60)
    draft = ask(researcher, "You are a python programer.", f"Create code for: {topic}")
    print("="*60)
    review = ask(critic, "You are a code reviewer.", f"Suggest improvements for this code:{draft}")
    print("="*60)
    final = ask(researcher, "You incorporate reviewer feedback.", f"Produce final code:{review}")
    print("== Final Plan ==", final)
