
from typing import List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data: (text, intent)
TRAINING_DATA: List[Tuple[str, str]] = [
    ("Order me a pizza", "order_pizza"),
    ("I want to order a pizza", "order_pizza"),
    ("Can you order me a pizza?", "order_pizza"),
    ("Please order a pizza", "order_pizza"),
    ("Get me a pizza", "order_pizza"),
    ("Hello", "greet"),
    ("Hi there", "greet"),
    ("Hey", "greet"),
    ("Good morning", "greet"),
]

INTENTS = [intent for _, intent in TRAINING_DATA]
TEXTS = [text for text, _ in TRAINING_DATA]

# Build a simple NLP pipeline: TF-IDF + Logistic Regression
MODEL: Pipeline = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=200)),
    ]
)

MODEL.fit(TEXTS, INTENTS)

CONFIDENCE_THRESHOLD = 0.45


def classify_intent(text: str) -> Tuple[str, float]:
    """Return (intent, confidence) for a given user message."""
    probs = MODEL.predict_proba([text])[0]
    classes = MODEL.classes_
    best_idx = int(probs.argmax())
    return classes[best_idx], float(probs[best_idx])


# Intent handlers

def handle_order_pizza(_: str) -> str:
    return "Order received. Your pizza is being prepared."


def handle_greet(_: str) -> str:
    return "Hello! You can ask me to order a pizza."


def handle_unknown(_: str) -> str:
    return "Sorry, I can only help with ordering a pizza right now."


HANDLERS = {
    "order_pizza": handle_order_pizza,
    "greet": handle_greet,
    "unknown": handle_unknown,
}


if __name__ == "__main__":
    print("NLP Bot (TF-IDF + Logistic Regression)")
    print("Type a message (or 'quit' to exit).\n")

    while True:
        user = input("> ").strip()
        if not user or user.lower() == "quit":
            break

        intent, confidence = classify_intent(user)
        if confidence < CONFIDENCE_THRESHOLD:
            intent = "unknown"

        response = HANDLERS[intent](user)
        print(f"Intent: {intent} (confidence={confidence:.2f}) | Response: {response}")
