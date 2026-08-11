# PII Detection — Naive Regex (Baseline)

import re
from utils.ollama_client import chat

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
CREDIT_CARD = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def redact(text: str) -> str:
    text = EMAIL.sub("[EMAIL_REDACTED]", text)
    text = PHONE.sub("[PHONE_REDACTED]", text)
    text = CREDIT_CARD.sub("[CARD_REDACTED]", text)
    text = SSN.sub("[SSN_REDACTED]", text)
    return text


def scan_output_for_pii(llm_response: str) -> str:
    return redact(llm_response)


if __name__ == "__main__":
    sample = (
        "Email me at sachin@example.com and call +91 98765-43210. "
        "My SSN is 123-45-6789 and card 4111-1111-1111-1111. "
        "My name is Sachin Dixit and I live at 42 Elm Street."
    )

    print("Regex PII redaction:")
    print(redact(sample))

    # Regex misses names and addresses — that's the point
    print("\nLimitations (NOT caught by regex):")
    print("  Person name  : 'Sachin Dixit' still visible")
    print("  Address      : '42 Elm Street' still visible")

    print("\nLLM call with regex guard:")
    user_msg = "My email is sachin@example.com — summarise my account."
    safe_input = redact(user_msg)
    print(f"  User (redacted): {safe_input}")
    messages = [
        {"role": "system", "content": "Never reveal personal data."},
        {"role": "user", "content": safe_input},
    ]
    response = chat(messages, temperature=0.2)
    print(f"  LLM  (scanned) : {scan_output_for_pii(response)}")
