def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n... [TRUNCATED] ...\n" + text[-tail:]


def build_prompt(system_prompt: str, context: str, user_query: str, max_input_chars: int, reserve_output_chars: int) -> str:
    budget_for_input = max(0, max_input_chars - reserve_output_chars)

    fixed = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_query}\n\n"
    remaining = max(0, budget_for_input - len(fixed))

    compact_context = truncate_middle(context, remaining)

    return f"SYSTEM:\n{system_prompt}\n\nCONTEXT:\n{compact_context}\n\nUSER:\n{user_query}"


def estimate_chars(prompt: str) -> int:
    return len(prompt)


if __name__ == "__main__":
    system_prompt = "You are a production assistant. Answer in 5 bullets, max 12 words each."
    user_query = "Give optimization recommendations for an agent handling support tickets."

    long_context = "\n".join(
        [f"Doc line {i}: detailed background data and historical decision logs." for i in range(1, 800)]
    )

    # Tweak these two numbers and watch how much context survives truncation.
    max_input_chars = 5000
    reserve_output_chars = 800
    prompt = build_prompt(system_prompt, long_context, user_query, max_input_chars, reserve_output_chars)

    print(f"Original context chars: {len(long_context)}")
    print(f"Final prompt chars    : {estimate_chars(prompt)}")
    print("\nPrompt preview:\n")
    print(prompt[:1200])
