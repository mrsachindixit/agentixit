
import re

# Sample training-like phrases (small set) for a pizza ordering intent
EXAMPLE_UTTERANCES = [
    "Order me a pizza",
    "I want to order a pizza",
    "Can you order me a pizza?",
    "Please order a pizza",
    "Get me a pizza",
]

INTENTS = {
    "order_pizza": {
        "patterns": [
            r"\border\b.*\bpizza\b",
            r"\bpizza\b.*\border\b",
            r"\bget\b.*\bpizza\b",
            r"\bwant\b.*\bpizza\b",
        ]
    },
    "greet": {
        "patterns": [
            r"\bhello\b",
            r"\bhi\b",
            r"\bhey\b",
        ]
    },
}

def detect_intent(text):
    """Return intent name based on regex pattern matching."""
    text = text.lower()
    for intent, data in INTENTS.items():
        for pattern in data["patterns"]:
            if re.search(pattern, text):
                return intent
    return "unknown"

# Intent handlers

def handle_order_pizza(text):
    """Simulate placing a pizza order."""
    return "Order received. Your pizza is being prepared."

def handle_greet(text):
    return "Hello! You can ask me to order a pizza."

def handle_unknown(text):
    return "Sorry, I can only help with ordering a pizza right now."

HANDLERS = {
    "order_pizza": handle_order_pizza,
    "greet": handle_greet,
    "unknown": handle_unknown,
}

if __name__ == "__main__":
    print("Example utterances for training:")
    for s in EXAMPLE_UTTERANCES:
        print(f"- {s}")

    print("\nType a message (or 'quit' to exit):")
    while True:
        user = input("> ").strip()
        if not user or user.lower() == "quit":
            break
        intent = detect_intent(user)
        response = HANDLERS[intent](user)
        print(f"Intent: {intent} | Response: {response}")
