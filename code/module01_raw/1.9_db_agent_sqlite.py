

import sqlite3
import json
import requests
import re

DB_FILE = "sample.db"

def init_db():
    """Initialize SQLite database with sample employee data."""
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary INT)")
    cur.execute("DELETE FROM employees")
    cur.executemany("INSERT INTO employees (name, dept, salary) VALUES (?, ?, ?)", [
        ("Asha", "Engineering", 2200000),
        ("Ravi", "Engineering", 1800000),
        ("Neha", "HR", 1200000),
        ("Ishaan", "Sales", 1500000),
    ])
    conn.commit()
    conn.close()

def safe_sql(sql: str) -> bool:
    """
    Guardrail: Only allow SELECT queries.
    Reject dangerous operations (DROP, DELETE, INSERT, UPDATE).
    """
    sql = sql.strip().lower()
    
    # Only SELECT allowed
    if not sql.startswith("select"):
        return False
    
    # Block dangerous keywords
    forbidden = [";", "--", "drop", "delete", "update", "insert", "alter"]
    return not any(tok in sql for tok in forbidden)

def run_sql(sql: str):
    """
    Execute a query if it passes safety checks.
    Returns results as {columns: [...], rows: [...]}
    """
    if not safe_sql(sql):
        return "Rejected: Only safe SELECTs allowed"
    
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    conn.close()
    
    return {"columns": cols, "rows": rows}


url = "http://localhost:11434/api/chat"
payload = {
    "model": "llama3.1:latest",
    "stream": False
}

if __name__ == "__main__":
    init_db()
    
    def ask_agent(task):
        """
        Two-step agent:
        Step 1: Generate safe SQL from natural language
        Step 2: Execute and summarize
        """
        print(f"User : {task}")
        print(f"— Step 1 : Asking assistant to generate SQL\n")
        schema = "Table employees(id, name, dept, salary)"
        
        # Step 1: Ask LLM to generate SQL
        planner = f"""
You can query SQLite. Use this schema: {schema}
Propose a SELECT query in strict JSON: {{"sql": "<QUERY>"}}
Only SELECT, limit to 50 rows if needed.
Output the JSON in ```json ... ``` code block.
Do not add any explanations, only JSON.

Example:
Task: List all employees in HR department.
```json
{{"sql": "SELECT name FROM employees WHERE dept='HR'"}}
```

Task: {task}
"""
        
        messages = [
            {
                "role": "system",
                "content": "You write safe SQL for SQLite. Generate a safe SQL query in JSON format. Proceed step by step."
            },
            {
                "role": "user",
                "content": planner
            }
        ]
        
        payload["messages"] = messages
        response = requests.post(url, json=payload)
        model_response_text = response.json().get("message", {}).get("content", "")
        # print("Step 1 - Generated SQL:\n", model_response_text)
        
        # Parse JSON from response
        match = re.search(r'(?s)```json\s*(\{.*?\})\s*```', model_response_text)
        if match:
            model_response_text = match.group(1)
        
        try:
            sql = json.loads(model_response_text)["sql"]
            # print(f"Parsed SQL: {sql}\n")
        except Exception as e:
            print(f"Parse failed ({e}), using fallback\n")
            sql = "SELECT * FROM employees LIMIT 5"
        
        # Step 2: Execute SQL
        print(f"— Step 2 : Executing SQL : {sql}\n")
        result = run_sql(sql)
        # print(f"Result :\n{result}\n")
        
        # Step 3: Ask LLM to summarize
        print(f"— Step 3 : Summarizing SQL result\n")
        messages = [
            {"role": "system", "content": f"You need to answer the user's query based on data retrieved from SQL database.\n Summarize the result of this query in 2-3 sentences.\n Only summarize the results don't mention anything about attributes not returned in sql query"},
            {"role": "user", "content": f"User Query : {task} \n SQL query result: {result}"}
        ]
        
        payload["messages"] = messages
        response = requests.post(url, json=payload)
        summary = response.json().get("message", {}).get("content", "")
        print(f"Assistant : {summary}\n")
        print("=" * 60 + "\n")

    # Test cases
    ask_agent("List Engineering employees with their salaries")
    ask_agent("How many HR employees are there?")
    ask_agent("Find the highest paid employee")


