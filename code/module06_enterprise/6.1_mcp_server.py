
"""
Pls see mcp sdk file for study , this file is for
acacdmic completness ,shows innerworkings of mcp without
the annotations .

"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import uvicorn
import time

app = FastAPI(title="MCP Server with Tool Call Agent")


# ==================== CORE MCP TEACHING CODE ====================


def get_weather(city: str) -> str:
    temps = {"bogotá": "12°C", "new york": "18°C", "london": "8°C", "tokyo": "22°C"}
    temp = temps.get(city.lower(), "15°C")
    return f"{city}: {temp}, Partly cloudy"


def get_pincode(city: str) -> str:
    pincodes = {"bogotá": "110111", "new york": "10001", "london": "SW1A1AA", "tokyo": "100-0001"}
    pincode = pincodes.get(city.lower(), "000000")
    return f"{city}: {pincode}"


TOOLS = {
    "get_weather": get_weather,
    "get_pincode": get_pincode,
}

TOOL_SCHEMAS = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    "get_pincode": {
        "name": "get_pincode",
        "description": "Get postal code for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}

PROMPT_SCHEMAS = {
    "city_weather_prompt": {
        "name": "city_weather_prompt",
        "description": "Prompt template to ask weather by city",
        "template": "What is the weather in {city}?",
        "input_variables": ["city"],
    },
    "city_pincode_prompt": {
        "name": "city_pincode_prompt",
        "description": "Prompt template to ask pincode by city",
        "template": "What is the pincode for {city}?",
        "input_variables": ["city"],
    },
}

RESOURCE_CATALOG = [
    "agent/tools",
    "agent/prompts",
    "tool/weather",
    "tool/pincode",
]


def list_tools() -> list[Dict[str, Any]]:
    return [
        {
            "name": schema["name"],
            "description": schema["description"],
            "parameters": schema.get("parameters", {}),
        }
        for schema in TOOL_SCHEMAS.values()
    ]


def list_prompts() -> list[Dict[str, Any]]:
    return list(PROMPT_SCHEMAS.values())


class Envelope(BaseModel):
    id: str
    type: str
    resource: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = {}


@app.post("/mcp/invoke")
async def invoke(envelope: Envelope):
    """MCP invoke endpoint.

    Resources:
    - "agent/tools": List available tools
    - "agent/prompts": List demo prompt templates
    - "tool/weather": Get weather for a city
    - "tool/pincode": Get pincode for a city
    """
    start = time.time()
    resource = envelope.resource
    payload = envelope.payload or {}

    if resource == "agent/tools":
        body = {"status": "ok", "result": list_tools()}

    elif resource == "agent/prompts":
        body = {"status": "ok", "result": list_prompts()}

    elif resource == "tool/weather":
        city = payload.get("city", "")
        body = {"status": "ok", "result": get_weather(city)} if city else {"status": "error", "error": "Missing 'city' parameter"}

    elif resource == "tool/pincode":
        city = payload.get("city", "")
        body = {"status": "ok", "result": get_pincode(city)} if city else {"status": "error", "error": "Missing 'city' parameter"}

    else:
        body = {"status": "error", "error": f"unknown resource: {resource}"}

    elapsed = time.time() - start
    return {
        "id": envelope.id,
        "type": "response",
        "resource": envelope.resource,
        "metadata": {"elapsed": elapsed},
        "payload": body,
    }


# ==================== OPTIONAL METADATA ENDPOINT ====================
@app.get("/mcp/catalog")
async def mcp_catalog():
    """Teaching-friendly metadata endpoint for tools/prompts/resources."""
    return {
        "server": {
            "name": "MCP Server with Tool Call Agent",
            "version": "1.2.0",
            "teaching_demo": True,
            "annotations_note": "Decorator-based annotations are in 6.3_mcp_sdk_server.py",
            "disclaimer": "This web inspector is for classroom/demo use. Production integrations usually rely on MCP clients/SDK transports.",
        },
        "resources": RESOURCE_CATALOG,
        "tools": list_tools(),
        "prompts": list_prompts(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
