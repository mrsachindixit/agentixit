import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ollama_client import chat


PROPOSERS = [
    ("formal",   "You are a precise academic. Prefer formal terminology and structured bullet points."),
    ("plain",    "You are a friendly explainer. Use plain language and analogies a non-expert would get."),
    ("skeptic",  "You are a critical engineer. List concrete failure modes and caveats."),
]

AGGREGATOR_SYS = (
    "You are the synthesiser. You will see N independent answers to the same question. "
    "Produce ONE final answer that:\n"
    "  - keeps facts agreed by 2+ proposers\n"
    "  - drops claims unique to one proposer (likely hallucination)\n"
    "  - explicitly notes any disagreements\n"
    "Be concise (under 120 words)."
)


def propose(name: str, system: str, question: str) -> tuple[str, str]:
    answer = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        temperature=0.3,
    )
    return name, answer.strip()


def aggregate(question: str, proposals: list[tuple[str, str]]) -> str:
    bundle = "\n\n".join(f"--- {name} ---\n{ans}" for name, ans in proposals)
    return chat(
        [
            {"role": "system", "content": AGGREGATOR_SYS},
            {"role": "user",   "content": f"Question: {question}\n\nProposals:\n{bundle}"},
        ],
        temperature=0.0,
    ).strip()


def moa(question: str) -> tuple[str, list[tuple[str, str]]]:
    with ThreadPoolExecutor(max_workers=len(PROPOSERS)) as ex:
        proposals = list(ex.map(lambda p: propose(*p, question), PROPOSERS))
    final = aggregate(question, proposals)
    return final, proposals


if __name__ == "__main__":
    question = "Why might my agent's tool calls succeed in dev but time-out in production, and what should I check first?"

    final, proposals = moa(question)

    print("=" * 70)
    for name, ans in proposals:
        print(f"\n[{name}]\n{ans}")
    print("=" * 70)
    print("\n[AGGREGATED FINAL]\n" + final)

