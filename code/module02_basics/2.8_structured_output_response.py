import json
import os
import requests
from pydantic import BaseModel, Field, ValidationError
from typing import Literal


# ---- target schema --------------------------------------------------------
class TicketTriage(BaseModel):
    category: Literal["billing", "tech", "account", "spam"]
    urgency: int = Field(ge=1, le=5)
    summary: str = Field(min_length=5, max_length=140)
    needs_human: bool


SYSTEM = (
    "You triage support tickets. Return ONLY this JSON shape:\n"
    + json.dumps(TicketTriage.model_json_schema(), indent=2)
)


# ---- Approach 1 : raw Ollama with format=json ------------------------------
def triage_ollama(ticket: str) -> TicketTriage:
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.1:latest",
            "format": "json",          # <-- key flag: forces JSON output
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": ticket},
            ],
        },
        timeout=120,
    )
    r.raise_for_status()
    raw = r.json()["message"]["content"]
    return TicketTriage.model_validate_json(raw)   # crashes if schema violated


# ---- Approach 2 : LangChain with_structured_output ------------------------
def triage_langchain(ticket: str) -> TicketTriage:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="llama3.1:latest", temperature=0)
    structured = llm.with_structured_output(TicketTriage)        # <-- one line
    return structured.invoke(f"{SYSTEM}\n\nTicket: {ticket}")


# ---- Approach 3 : provider-agnostic retry-with-repair ---------------------
def triage_retry(ticket: str, max_attempts: int = 3) -> TicketTriage:
    """Works with ANY chat model. Uses validation error as the repair hint."""
    from utils.ollama_client import chat
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        prompt = ticket if attempt == 1 else (
            f"Your previous response failed schema validation: {last_error}\n"
            f"Output a corrected JSON. Ticket: {ticket}"
        )
        raw = chat([
            {"role": "system", "content": SYSTEM + "\nReturn ONLY a JSON object, no prose."},
            {"role": "user", "content": prompt},
        ])
        # tolerate extra prose around JSON
        start, end = raw.find("{"), raw.rfind("}")
        try:
            return TicketTriage.model_validate_json(raw[start:end + 1])
        except (ValidationError, ValueError) as e:
            last_error = str(e)[:300]
            print(f"  [attempt {attempt}] parse error -> retrying")
    raise RuntimeError(f"Schema enforcement failed after {max_attempts} attempts: {last_error}")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    ticket = "I was billed twice for the Pro plan in May and the second charge is from a country I have never visited."

    print("--- Approach 1: Ollama format=json ---")
    print(triage_ollama(ticket).model_dump())

    print("\n--- Approach 2: LangChain with_structured_output ---")
    try:
        print(triage_langchain(ticket).model_dump())
    except Exception as e:
        print(f"(skipped: {e})")

    print("\n--- Approach 3: retry-with-repair (provider-agnostic) ---")
    print(triage_retry(ticket).model_dump())

    # Why this matters: downstream code (router, dashboard, alerting) NEVER touches a
    # free-text string. Parse failures crash loud at the boundary, not silently later.
