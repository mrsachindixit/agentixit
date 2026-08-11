import asyncio
import os

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

import requests

SERVER = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


async def run_agent(user_message):
    async with streamable_http_client(SERVER) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available Tools in mcp server :", [t.name for t in tools.tools])

            # result = await session.call_tool("get_weather", {"city": "London"})
            # print("Weather:", result.content[0].text if result.content else result.structuredContent)

            # result = await session.call_tool("get_pincode", {"city": "Tokyo"})
            # print("Pincode:", result.content[0].text if result.content else result.structuredContent)

            # resources = await session.list_resources()
            # print("Resources:", [str(r.uri) for r in resources.resources])

            # content = await session.read_resource("data://cities")
            # block = content.contents[0]
            # print("Cities:", block.text if isinstance(block, types.TextResourceContents) else block)

            # content = await session.read_resource("data://city/Tokyo")
            # block = content.contents[0]
            # print("Detail:", block.text if isinstance(block, types.TextResourceContents) else block)

            # prompts = await session.list_prompts()
            # print("Prompts:", [p.name for p in prompts.prompts])

            # prompt = await session.get_prompt("weather_prompt", {"city": "Tokyo"})
            # print("Prompt text:", prompt.messages[0].content.text)

            # prompt = await session.get_prompt("pincode_prompt", {"city": "London"})
            # print("Prompt text:", prompt.messages[0].content.text)

            # Convert FastMCP tools to Ollama Tool format
            ollama_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
                for tool in tools.tools
            ]

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
                "tools": ollama_tools,
                "stream": False
            }).json()

            # 3. Check if the model wants to use the tool
            if "tool_calls" in res["message"]:
                tool_call = res["message"]["tool_calls"][0]
                city = tool_call["function"]["arguments"]["city"]
                
                print(f"— Tool Call Triggered for getting weather of {city}")
                
                # Execute the actual weather fetch
                result = await session.call_tool("get_weather", {"city": city})
                weather_info = result.content[0].text if result.content else result.structuredContent
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
    answer = asyncio.run(run_agent(user_question))
    print(f"Final Answer : {answer}")
