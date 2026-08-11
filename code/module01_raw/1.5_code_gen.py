

import subprocess
import sys
import tempfile

from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:latest", base_url="http://localhost:11434/", temperature=0.1)

def step_requirements(reqs):
    """Business logic: Parse requirements (placeholder)."""
    return f"Clarified requirements: {reqs}"

def step_design(reqs):
    """Business logic: Create high-level design (placeholder)."""
    return f"High-level design for: {reqs}"

def step_impl(design):
    """Use LLM to write Python skeleton code from design."""
    messages = [
        {"role": "system", "content": "Write concise Python skeleton code for the described design."},
        {"role": "user", "content": design}
    ]
    return llm.invoke(messages).content

def generate_code(prompt: str) -> str:
    """
    Generate runnable Python code from a natural language prompt.
    
    Args:
        prompt: Natural language description of code to generate
        
    Returns:
        Generated Python code (no markdown, ready to execute)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python code generator. Output ONLY runnable Python code. "
                "Do not include markdown fences, explanations, or comments."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return llm.invoke(messages).content

def is_safe_code(code: str) -> bool:
    """
    Naive safety check to block common dangerous patterns.
    
    Args:
        code: Python code to check
        
    Returns:
        True if code appears safe, False if dangerous patterns found
        
    Note: This is simplistic. Real security needs deeper analysis (AST parsing).
    """
    blocked = ["import ", "__", "open(", "exec(", "eval(", "os.", "sys.", "subprocess"]
    return not any(b in code for b in blocked)

def run_in_sandbox(code: str) -> str:
    """
    Run code in a subprocess with timeout and capture output.
    
    Args:
        code: Python code to execute
        
    Returns:
        stdout output, error message, or timeout notice
    """
    if not is_safe_code(code):
        return "Sandbox blocked execution: unsafe patterns detected."

    # Write code to temp file
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Execute in subprocess (isolated from main process)
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=3,  # 3-second timeout
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        
        if err:
            return f"Error:\n{err}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Sandbox timeout: execution took too long."


if __name__ == "__main__":
    # Example 1: Three-step workflow (requirements → design → implementation)
    print("=== EXAMPLE 1: Design Workflow ===\n")
    reqs = step_requirements("Agent reads a DB, runs a safe SELECT, explains results.")
    design = step_design(reqs)
    code = step_impl(design)
    print("Generated Skeleton:\n")
    print(code)

    # Example 2: Generate code from prompt and execute it
    print("\n=== EXAMPLE 2: Code Generation + Sandbox ===\n")
    prompt = (
        "Write a Python script that prints the first 5 Fibonacci numbers, "
        "one per line."
    )
    generated = generate_code(prompt)
    print("Generated Code:\n")
    print(generated)

    print("\nExecution Output:\n")
    output = run_in_sandbox(generated)
    print(output)


    print("\n== Sandbox Output ==\n")
    print(output)