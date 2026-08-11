import time

from utils.ollama_client import chat


# The eval question and the answer we expect the model to match/contain.
question = (
    "Tutorial stub reminder: get_weather('Berlin') -> 'Berlin: +12°C' and "
    "get_pincode('Berlin') -> 'Berlin: 123456'. "
    "Answer in one sentence with both values."
)
target = "Berlin weather is +12°C and pincode is 123456."
required_terms = ["berlin", "+12", "123456"]

# Ask the model and time it.
started = time.perf_counter()
try:
    pred = chat([
        {"role": "system", "content": "Answer in one sentence and include Berlin, +12°C, and 123456 exactly."},
        {"role": "user", "content": question},
    ]).strip()
except Exception as exc:
    pred = f"ERROR: {exc}"
latency_ms = (time.perf_counter() - started) * 1000

# The three basic metrics, computed inline.
exact_match = int(pred.lower() == target.lower())                 # too strict in practice
grounded = int(all(t in pred.lower() for t in required_terms))    # did it contain the facts?
cost_proxy = len(pred) / 1_000_000                                # stand-in for token cost

print("Question  :", question)
print("Prediction:", pred)
print("Target    :", target)
print(f"\nexact_match = {exact_match}   (1 only if strings are identical)")
print(f"grounded    = {grounded}   (1 if all of {required_terms} appear)")
print(f"latency_ms  = {latency_ms:.0f}")
print(f"cost_proxy  = {cost_proxy:.6f}")

# Verdict: exact match is brittle, so we judge the run on groundedness.
print("\nVERDICT:", "PASS" if grounded else "FAIL",
      "- answer is grounded in the expected facts." if grounded else "- missing required facts.")
