 

import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import { ToolMessage } from "@langchain/core/messages";
import { z } from "zod";

const OLLAMA_BASE  = process.env.OLLAMA_BASE  ?? "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL ?? "llama3.1:latest";
const model = new ChatOllama({ baseUrl: OLLAMA_BASE, model: OLLAMA_MODEL, temperature: 0 });
const getWeather = tool(
  async ({ city }) => `${city}: +12°C, partly cloudy`,
  {
    name: "get_weather",
    description: "Return the current temperature for a given city",
    schema: z.object({
      city: z.string().describe("City name, e.g. 'London'"),
    }),
  }
);
const modelWithTools = model.bindTools([getWeather]);
async function runAgent(userMessage) {
  const messages = [{ role: "user", content: userMessage }];
  console.log("Calling model...");
  const aiMsg = await modelWithTools.invoke(messages);
  messages.push(aiMsg);

  for (const toolCall of aiMsg.tool_calls ?? []) {
    if (toolCall.name === "get_weather") {
      console.log(`Tool: ${JSON.stringify(toolCall.args)}`);
      const toolResult = await getWeather.invoke(toolCall.args);
      console.log(`Result: ${toolResult}`);
      messages.push(new ToolMessage({ content: toolResult, name: toolCall.name, tool_call_id: toolCall.id }));
    }
  }

  console.log("Finalizing...");
  const finalMsg = await modelWithTools.invoke(messages);
  return finalMsg.content;
}

const userQuestion = "What's the weather in Bogotá in celsius?";
console.log("LangChain JS Tool Call");
console.log(`User: ${userQuestion}\n`);

runAgent(userQuestion)
  .then(answer => console.log(`Agent: ${answer}`))
  .catch(err   => { console.error("Error:", err.message); process.exit(1); });
