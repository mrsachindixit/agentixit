import os
import re
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import chat


PROBLEMS = [
    ("Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many balls does he have now?", 11),
    ("If 3 shirts take 1 hour to dry on a line, how many hours do 9 shirts take?", 1),  # parallel drying — trick
    ("A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?", 0.05),
]


def extract_number(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def zero_shot(q: str) -> str:
    return chat([{"role": "user", "content": q + " Give the final number only."}], temperature=0)


def zero_shot_cot(q: str) -> str:
    return chat([
        {"role": "system", "content": "Reason step by step. End with: 'Final answer: <number>'."},
        {"role": "user",   "content": q},
    ], temperature=0)


FEW_SHOT_EXAMPLES = """
Q: A train leaves at 9:00 going 60 mph. How far at 11:30?
A: From 9:00 to 11:30 is 2.5 hours. 60 * 2.5 = 150. Final answer: 150

Q: Anna has 4 books. She gives away 1 and buys 3. How many books?
A: 4 - 1 + 3 = 6. Final answer: 6
""".strip()

def few_shot_cot(q: str) -> str:
    return chat([
        {"role": "system", "content": "Reason like the examples. End with 'Final answer: <number>'."},
        {"role": "user",   "content": FEW_SHOT_EXAMPLES + f"\n\nQ: {q}\nA:"},
    ], temperature=0)


def self_consistency(q: str, samples: int = 5) -> tuple[float | None, list]:
    answers = []
    for _ in range(samples):
        reply = chat([
            {"role": "system", "content": "Reason like the examples. End with 'Final answer: <number>'."},
            {"role": "user",   "content": FEW_SHOT_EXAMPLES + f"\n\nQ: {q}\nA:"},
        ], temperature=0.7)
        answers.append(extract_number(reply))
    voted = Counter(a for a in answers if a is not None).most_common(1)
    return (voted[0][0] if voted else None), answers


if __name__ == "__main__":
    correct = {"zero": 0, "cot": 0, "few": 0, "sc": 0}
    for q, gold in PROBLEMS:
        print(f"\nQ: {q}")
        print(f"  gold = {gold}")

        a_zero = extract_number(zero_shot(q))
        a_cot  = extract_number(zero_shot_cot(q))
        a_few  = extract_number(few_shot_cot(q))
        a_sc, samples = self_consistency(q, samples=5)

        print(f"  zero-shot       -> {a_zero}")
        print(f"  zero-shot CoT   -> {a_cot}")
        print(f"  few-shot CoT    -> {a_few}")
        print(f"  self-consist.   -> {a_sc}   (samples: {samples})")

        for k, v in zip(["zero", "cot", "few", "sc"], [a_zero, a_cot, a_few, a_sc]):
            if v is not None and abs(v - gold) < 1e-2:
                correct[k] += 1

    n = len(PROBLEMS)
    print("\n=== accuracy summary ===")
    for k in ["zero", "cot", "few", "sc"]:
        print(f"  {k:>5}: {correct[k]}/{n}")

