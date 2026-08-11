
import requests
from typing import Dict, Any

DEFAULT_BASE = "http://127.0.0.1:8002"


def register(name: str, base: str = DEFAULT_BASE) -> Dict[str, Any]:
    r = requests.post(f"{base}/agents/register", json={"name": name}, timeout=10)
    r.raise_for_status()
    return r.json()


def send(sender_id: str, recipient_id: str, content: str, base: str = DEFAULT_BASE) -> Dict[str, Any]:
    payload = {"sender_id": sender_id, "recipient_id": recipient_id, "content": content}
    r = requests.post(f"{base}/agents/{sender_id}/send", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def inbox(agent_id: str, base: str = DEFAULT_BASE) -> Dict[str, Any]:
    r = requests.get(f"{base}/agents/{agent_id}/inbox", timeout=10)
    r.raise_for_status()
    return r.json()


if __name__ == '__main__':
    print("A2A client demo — register two agents, send messages, and poll inboxes")
    a1 = register("Agent-One")
    a2 = register("Agent-Two")
    print("Registered:", a1, a2)

    # send from a1 to a2
    send(a1['agent_id'], a2['agent_id'], "Hello from Agent-One to Agent-Two")
    send(a2['agent_id'], a1['agent_id'], "Acknowledged — Agent-Two here")

    print("Agent-One inbox:", inbox(a1['agent_id']))
    print("Agent-Two inbox:", inbox(a2['agent_id']))
