import os
import time

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def get_refund_policy(plan_name: str) -> str:
    return f"{plan_name} plan refund policy: full refund within 7 days, prorated refund within 30 days."


def get_invoice_status(invoice_id: str) -> str:
    return f"Invoice {invoice_id}: paid on 2026-05-12."


# The whole lesson: same agent, the only difference is whether caching middleware is attached.
def build_agent(use_cache):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY before running this example.")

    model = ChatAnthropic(model="claude-sonnet-4-5", api_key=api_key, temperature=0, max_tokens=700)
    middleware = [AnthropicPromptCachingMiddleware(ttl="1h", min_messages_to_cache=2)] if use_cache else []

    return create_agent(
        model=model,
        tools=[get_refund_policy, get_invoice_status],
        system_prompt=(
            "You are an enterprise support assistant for a subscription SaaS company. "
            "Always answer in three parts: summary, evidence used, next action. "
            "Use tools when refund policy or invoice status is requested."
        ),
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


# Two turns share the same long system prompt + history, so turn 2 is where caching pays off.
q1 = "I am on the Pro plan. Tell me the refund policy and also check invoice INV-2048."
q2 = "Based on that, what should I do next if I still want a refund today?"

for use_cache in [False, True]:
    label = "Cache ON" if use_cache else "Cache OFF"
    agent = build_agent(use_cache)
    thread = {"configurable": {"thread_id": label}}

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    for turn, question in [(1, q1), (2, q2)]:
        started = time.perf_counter()
        result = agent.invoke({"messages": [{"role": "user", "content": question}]}, thread)
        elapsed = time.perf_counter() - started

        # usage_metadata is the standard LangChain field; cache hits/writes live in input_token_details.
        usage = getattr(result["messages"][-1], "usage_metadata", None) or {}
        details = usage.get("input_token_details", {})
        print(
            f"turn {turn}: {elapsed:5.2f}s  "
            f"input={usage.get('input_tokens', 0):<5} output={usage.get('output_tokens', 0):<5} "
            f"cache_read={details.get('cache_read', 0):<5} cache_write={details.get('cache_creation', 0)}"
        )

print("\nWatch turn 2 under 'Cache ON': cache_read jumps and input_tokens (billed) drops.")
