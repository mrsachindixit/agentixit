

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.tools import tool

llm = ChatOllama(
    model="llama3.1:latest", base_url='http://localhost:11434', temperature=0)


@tool
def add(q: str) -> str:
    """Add two numbers provided in the format 'a b' and return the result as a string.
    Args:
    q (str): A string containing two numbers separated by a space, e.g., '3 5'.

    Returns:
    str: The sum of the two numbers as a string.

    """
    print(f"— Tool Call Triggered for add: {q}")
    a, b = q.split()
    result = str(float(a) + float(b))
    print(f"— Tool Result for add: {result}")
    return result


@tool
def multiply(q: str) -> str:
    """Multiply two numbers provided in the format 'a b' and return the result as a string.
    Args:
    q (str): A string containing two numbers separated by a space, e.g., '3 5'.

    Returns:
    str: The product of the two numbers as a string.

    """
    print(f"— Tool Call Triggered for multiply: {q}")
    a, b = q.split()
    result = str(float(a) * float(b))
    print(f"— Tool Result for multiply: {result}")
    return result


agent = create_agent(
    model=llm,
    tools=[add, multiply],

    system_prompt="You are a helpful assistant that can perform basic arithmetic operations like addition and multiplication. You will receive questions in the format 'Compute a * b, then add c to the result. Show steps briefly.' Your task is to parse the question, perform the calculations using the provided tools, and return the final answer along with a brief explanation of the steps taken." \
    "Example " \
    "Question: Compute 12 * 9, then add 15 to the result. Show steps briefly." \
    "Answer: First, I will multiply 12 and 9 using the multiply tool. The result is 108. Then, I call the add tool to add with 108 15. The final answer is 123."
)
def run_agent(user_message):
    print("— Asking Ollama...")
    ai_msg = agent.invoke(
        {
            "messages": [{"role": "user", "content": user_message}]
        }
    )
    return ai_msg.get("messages")[-1].content

if __name__ == "__main__":
    user_question = "Compute 11 * 9, then add 10 to the result. Show steps briefly."
    print(f"User: {user_question}\n")
    answer = run_agent(user_question)
    print(f"Agent: {answer}")

    # user_question = "Multiply 8 * 32, then add 20 to the result. Show steps briefly."
    # print(f"User: {user_question}\n")
    # answer = run_agent(user_question)
    # print(f"Agent: {answer}")
