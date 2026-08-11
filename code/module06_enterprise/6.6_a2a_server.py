
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import uuid
import uvicorn

app = FastAPI(title="A2A Example Server")

# In-memory storage for demo purposes
AGENTS: Dict[str, Dict[str, Any]] = {}
INBOXES: Dict[str, List[Dict[str, Any]]] = {}


class RegisterRequest(BaseModel):
    name: str


class Message(BaseModel):
    sender_id: str
    recipient_id: str
    content: str
    metadata: Dict[str, Any] = {}


@app.post('/agents/register')
def register_agent(req: RegisterRequest):
    agent_id = str(uuid.uuid4())
    AGENTS[agent_id] = {"id": agent_id, "name": req.name}
    INBOXES[agent_id] = []
    return {"status": "ok", "agent_id": agent_id}


@app.post('/agents/{agent_id}/send')
def send_message(agent_id: str, msg: Message):
    if agent_id != msg.sender_id:
        raise HTTPException(status_code=400, detail="sender_id mismatch with path agent_id")
    if msg.recipient_id not in INBOXES:
        raise HTTPException(status_code=404, detail="recipient not found")
    envelope = {
        "id": str(uuid.uuid4()),
        "from": msg.sender_id,
        "to": msg.recipient_id,
        "content": msg.content,
        "metadata": msg.metadata,
    }
    INBOXES[msg.recipient_id].append(envelope)
    return {"status": "delivered", "envelope_id": envelope["id"]}


@app.get('/agents/{agent_id}/inbox')
def get_inbox(agent_id: str):
    if agent_id not in INBOXES:
        raise HTTPException(status_code=404, detail="agent not found")
    # return a shallow copy
    return {"status": "ok", "messages": INBOXES[agent_id]}


@app.post('/agents/{agent_id}/inbox/clear')
def clear_inbox(agent_id: str):
    if agent_id not in INBOXES:
        raise HTTPException(status_code=404, detail="agent not found")
    INBOXES[agent_id].clear()
    return {"status": "ok"}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8002)
