
import requests
import uuid
from typing import Dict, Any

DEFAULT_URL = "http://127.0.0.1:8001/mcp/invoke"


def make_envelope(resource: str, payload: Dict[str, Any], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "type": "invoke",
        "resource": resource,
        "payload": payload,
        "metadata": metadata or {},
    }


def invoke(resource: str, payload: Dict[str, Any], url: str = DEFAULT_URL):
    env = make_envelope(resource, payload)
    r = requests.post(url, json=env, timeout=30)
    r.raise_for_status()
    return r.json()


def print_response(title: str, resp: Dict):
    """Pretty print response"""
    print(f"\n{'='*60}")
    print(f"- {title}")
    print(f"{'='*60}")
    import json
    print(json.dumps(resp.get('payload', resp), indent=2))


if __name__ == "__main__":
    print("Raw MCP Envelope Client Demo\n")
    print("This demonstrates direct invoke envelopes for 6.1_mcp_server.py\n")

    print("\n" + "DISCOVERY ENDPOINTS".center(60))
    resp = invoke("agent/tools", {})
    print_response("Available Tools in Agent", resp)

    resp = invoke("agent/prompts", {})
    print_response("Available Prompts in Agent", resp)

    print("\n" + "DIRECT TOOL CALLS".center(60))
    
    resp = invoke("tool/weather", {"city": "Bogotá"})
    print_response("Direct Tool: Get Weather for Bogotá", resp)
    
    resp = invoke("tool/pincode", {"city": "London"})
    print_response("Direct Tool: Get Pincode for London", resp)
