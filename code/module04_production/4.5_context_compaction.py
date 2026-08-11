import json
import urllib.request


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def count_chars(messages) -> int:
    return sum(len(m["content"]) for m in messages if isinstance(m.get("content"), str))


def chat_once(messages) -> str:
    payload = json.dumps(
        {"model": "llama3.1:latest", "messages": messages, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return str(data["message"]["content"])


def compact_history(messages, keep_last_turns: int = 4):
    if len(messages) <= keep_last_turns + 1:
        return messages

    system = messages[0]
    old = messages[1:-keep_last_turns]
    recent = messages[-keep_last_turns:]

    old_text = "\n".join(m["content"] for m in old if isinstance(m.get("content"), str))
    summary_prompt = (
        "Summarize this chat history into max 6 bullets with only enduring facts, decisions, and constraints:\n\n"
        + old_text
    )
    summary = chat_once([{"role": "user", "content": summary_prompt}])
    summary_msg = {"role": "user", "content": f"Context summary:\n{summary}"}

    return [system, summary_msg, *recent]


def demo_messages():
    msgs = [{"role": "system", "content": "You are a concise production assistant."}]
    for i in range(1, 11):
        msgs.append({"role": "user", "content": f"User turn {i}: discuss architecture item {i} with details and constraints."})
        msgs.append({"role": "assistant", "content": f"Assistant turn {i}: acknowledged and proposed option {i}."})
    return msgs


if __name__ == "__main__":
    messages = demo_messages()

    before_chars = count_chars(messages)
    compacted = compact_history(messages, keep_last_turns=4)
    after_chars = count_chars(compacted)

    print(f"Messages before: {len(messages)} | chars: {before_chars}")
    print(f"Messages after : {len(compacted)} | chars: {after_chars}")
    print(f"Reduction      : {100 * (before_chars - after_chars) / max(before_chars, 1):.1f}%")

    question = {"role": "user", "content": "Given context, suggest next 3 actions."}
    final = chat_once([*compacted, question])
    print("\nAnswer:")
    print(final)
