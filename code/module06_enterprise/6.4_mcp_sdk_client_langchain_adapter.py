import asyncio
import os

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


async def run_agent(user_message: str) -> str:
    client = MultiServerMCPClient(
        {
            "weather-server": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            }
        }
    )

    tools = await client.get_tools()
    print("Available Tools in mcp server:", [tool.name for tool in tools])

    llm = ChatOllama(
        model="llama3.1:latest",
        base_url="http://localhost:11434",
        temperature=0,
    )

    agent = create_agent(model=llm, tools=tools)
    result = await agent.ainvoke(
        {
            "messages": [
                {"role": "system", "content": "You are a helpful, step-by-step assistant. You can consider data returned from tool is latest."},
                {"role": "user", "content": user_message}    
            ]
        }
    )

    final_message = result["messages"][-1]
    return str(final_message.content)


if __name__ == "__main__":
    user_question = "What's the weather in Bogotá, Colombia in celsius?"
    print(f"User: {user_question}\n")
    answer = asyncio.run(run_agent(user_question))
    print(f"Final Answer: {answer}")