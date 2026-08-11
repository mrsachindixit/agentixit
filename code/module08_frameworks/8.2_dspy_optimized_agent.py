import os
import ast
import dspy

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

lm = dspy.LM(
    model=f"ollama_chat/{OLLAMA_MODEL}",
    api_base=OLLAMA_BASE,
    api_key="",
    max_tokens=512,
)
dspy.configure(lm=lm)

def calculate(expression: str) -> str:
    try:
        node = ast.parse(expression, mode="eval")
        for n in ast.walk(node):
            if isinstance(n, (ast.Call, ast.Name)):
                return "Error: only arithmetic allowed"
        result = eval(compile(node, "<string>", "eval"), {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def explain_steps(expression: str) -> str:
    steps = [
        f"Expression: {expression}",
        "Step 1: Identify operations following PEMDAS/BODMAS order",
        "Step 2: Evaluate parentheses first (if any)",
        "Step 3: Handle exponents",
        "Step 4: Multiply / Divide left to right",
        "Step 5: Add / Subtract left to right",
        f"Step 6: Final answer = {calculate(expression)}",
    ]
    return "\n".join(steps)


TOOL_REGISTRY = {
    "calculate": calculate,
    "explain_steps": explain_steps,
}

class DecideTool(dspy.Signature):
    question: str = dspy.InputField(desc="user's math question")
    tool_name: str = dspy.OutputField(desc="either 'calculate' or 'explain_steps'")
    tool_arg: str = dspy.OutputField(desc="the arithmetic expression to evaluate, e.g. '2+3*4'")


class Summarize(dspy.Signature):
    question: str = dspy.InputField()
    tool_result: str = dspy.InputField(desc="raw output from the tool")
    answer: str = dspy.OutputField(desc="final natural-language answer for the user")


class CalculatorAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.decide = dspy.ChainOfThought(DecideTool)
        self.summarize = dspy.ChainOfThought(Summarize)

    def forward(self, question: str):
        decision = self.decide(question=question)
        tool_name = decision.tool_name.strip().lower()
        tool_arg = decision.tool_arg.strip()

        tool_fn = TOOL_REGISTRY.get(tool_name)
        if tool_fn is None:
            return dspy.Prediction(answer=f"Unknown tool '{tool_name}'. Choose calculate or explain_steps.")
        tool_result = tool_fn(tool_arg)

        summary = self.summarize(question=question, tool_result=tool_result)
        return dspy.Prediction(
            tool_name=tool_name,
            tool_arg=tool_arg,
            tool_result=tool_result,
            answer=summary.answer,
        )

TRAIN_EXAMPLES = [
    dspy.Example(
        question="What is 15 + 27?",
        tool_name="calculate",
        tool_arg="15 + 27",
    ).with_inputs("question"),
    dspy.Example(
        question="Calculate 100 divided by 4",
        tool_name="calculate",
        tool_arg="100 / 4",
    ).with_inputs("question"),
    dspy.Example(
        question="Show me the steps to solve 3 * (10 + 5)",
        tool_name="explain_steps",
        tool_arg="3 * (10 + 5)",
    ).with_inputs("question"),
    dspy.Example(
        question="Walk me through how 8**2 - 4 is evaluated",
        tool_name="explain_steps",
        tool_arg="8**2 - 4",
    ).with_inputs("question"),
    dspy.Example(
        question="Compute 2 ** 10",
        tool_name="calculate",
        tool_arg="2 ** 10",
    ).with_inputs("question"),
]


def tool_selection_metric(example, prediction, trace=None) -> float:
    name_ok = prediction.tool_name.strip().lower() == example.tool_name.strip().lower()
    arg_ok  = prediction.tool_arg.strip() == example.tool_arg.strip()
    return float(name_ok and arg_ok)


def optimize_agent() -> CalculatorAgent:
    agent = CalculatorAgent()
    optimizer = dspy.BootstrapFewShot(
        metric=tool_selection_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(agent, trainset=TRAIN_EXAMPLES)
    print("Optimization complete\n")
    return optimized

if __name__ == "__main__":
    print("=== DSPy Calculator ===")
    print()

    test_questions = [
        "What is 12 * 8 + 5?",
        "Explain step by step how to evaluate (2 + 3) * 4",
    ]

    print("--- Baseline ---")
    baseline = CalculatorAgent()
    for q in test_questions:
        r = baseline(question=q)
        print(f"  Q: {q}")
        print(f"     Tool: {r.tool_name}({r.tool_arg})  →  {r.tool_result}")
        print(f"     Answer: {r.answer}\n")

    print("--- Optimized ---")
    try:
        optimized = optimize_agent()
        for q in test_questions:
            r = optimized(question=q)
            print(f"  Q: {q}")
            print(f"     Tool: {r.tool_name}({r.tool_arg})  →  {r.tool_result}")
            print(f"     Answer: {r.answer}\n")
    except Exception as exc:
        print(f"  Optimization skipped: {exc}")
        print()
