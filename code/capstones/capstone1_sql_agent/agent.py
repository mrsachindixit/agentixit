import json
import sqlite3
import os
from typing import Any, Dict
import re

from utils.ollama_client import chat

FORBIDDEN = [';','--','drop','delete','update','insert','alter','attach','pragma','create','vacuum']
RISKY_PATTERNS = [
    (r"\bjoin\b", 2),
    (r"\bgroup\s+by\b", 2),
    (r"\bhaving\b", 2),
    (r"\bwith\b", 3),
    (r"\border\s+by\b", 1),
    (r"\bcount\s*\(", 1),
    (r"\bsum\s*\(", 1),
    (r"\bavg\s*\(", 1),
    (r"\bdistinct\b", 1),
]

def safe(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith('select') and not any(b in s for b in FORBIDDEN)


def risk_score(sql: str) -> tuple[int, list[str]]:
    s = sql.strip().lower()
    score = 1
    reasons = []
    for pat, weight in RISKY_PATTERNS:
        if re.search(pat, s):
            score += weight
            reasons.append(f"matched:{pat}")
    if " limit " not in f" {s} ":
        score += 2
        reasons.append("no_limit")
    if len(s) > 220:
        score += 1
        reasons.append("long_query")
    return score, reasons


class SQLiteAnalystAgent:
    """A small, teachable agent for querying SQLite databases safely.

    Features:
    - Schema inspection
    - Natural language -> safe SELECT generation (via Ollama chat)
    - Execute and return results
    - Explain query plan
    - Summarize results for end users
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def schema(self) -> Dict[str, Any]:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        table_names = [r[0] for r in cur.fetchall()]
        schema = {}
        for t in table_names:
            # columns and types
            cur.execute(f"PRAGMA table_info('{t}')")
            cols = [{'cid': r[0], 'name': r[1], 'type': r[2], 'notnull': r[3], 'dflt_value': r[4], 'pk': r[5]} for r in cur.fetchall()]
            # sample rows
            cur.execute(f"SELECT * FROM {t} LIMIT 3")
            sample = cur.fetchall()
            schema[t] = {'columns': cols, 'sample_rows': sample}
        conn.close()
        return schema

    def suggest_sql(self, nlq: str, examples: str = '') -> Dict[str, str]:
        """Ask the model to propose a safe SELECT query. Returns dict with 'sql' and 'explain'."""
        schema = self.schema()
        system = (
            "You are a SQL assistant that ONLY outputs valid JSON with keys: sql, explanation.\n"
            "Generate a single SQLite SELECT statement (no semicolons) that answers the user's request.\n"
            "You may use JOINs, GROUP BY, HAVING, ORDER BY, CTEs (WITH), and aggregate functions.\n"
            "Never produce DDL or DML (CREATE, INSERT, UPDATE, DELETE, DROP, PRAGMA, ATTACH).\n"
            "When appropriate, add a LIMIT to avoid huge results. Use the provided schema and sample rows to choose appropriate columns."
        )
        examples_block = (
            "Examples:\n"
            "1) NL: 'Total payroll per department in 2023' -> SQL: 'SELECT d.name AS dept, SUM(salary) AS total_payroll FROM employees e JOIN departments d ON e.dept_id=d.id GROUP BY d.name'\n"
            "2) NL: 'Who reports to Asha and their projects?' -> SQL: 'SELECT e.name AS employee, p.name AS project, ep.role FROM employees e JOIN employee_projects ep ON e.id=ep.employee_id JOIN projects p ON ep.project_id=p.id WHERE e.manager_id = (SELECT id FROM employees WHERE name=\'Asha\')'\n"
        )
        user = f"Schema: {json.dumps(schema)}\nUser request: {nlq}\n{examples_block}\nAdditional examples: {examples}"
        try:
            draft = chat([{"role":"system","content":system},{"role":"user","content":user}], temperature=0)
            # model should return JSON like {"sql": "SELECT ...", "explanation": "..."}
            # Attempt to parse JSON even if model wraps code fences
            text = draft.strip()
            if text.startswith("```"):
                # strip code fences
                parts = text.split('```')
                # take first non-empty part
                text = next((p for p in parts if p.strip()), text)
            parsed = json.loads(text)
            sql = parsed.get('sql','').strip()
            explain = parsed.get('explanation','')
        except Exception:
            # fallback: very simple heuristic
            sql = "SELECT * FROM employees LIMIT 50"
            explain = "Fallback query: return up to 50 employees."
        return {"sql": sql, "explanation": explain}

    def execute(self, sql: str, limit: int = 100, require_approval_above: int = 6, approved: bool = False) -> Dict[str, Any]:
        if not safe(sql):
            return {"error": "unsafe_sql"}
        score, reasons = risk_score(sql)
        if score >= require_approval_above and not approved:
            return {
                "error": "approval_required",
                "risk_score": score,
                "risk_reasons": reasons,
                "message": "Query looks complex/risky for classroom mode. Re-run with approval.",
            }
        conn = self._connect()
        cur = conn.cursor()
        # apply an upper-bound limit if missing
        s = sql.strip().lower()
        if 'limit' not in s:
            sql = sql.rstrip() + f" LIMIT {limit}"
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            conn.close()
            return {
                "columns": cols,
                "rows": rows,
                "rowcount": len(rows),
                "risk_score": score,
                "risk_reasons": reasons,
            }
        except Exception as e:
            conn.close()
            return {"error": "execution_error", "message": str(e)}

    def explain(self, sql: str) -> str:
        conn = self._connect()
        cur = conn.cursor()
        try:
            cur.execute(f"EXPLAIN QUERY PLAN {sql}")
            rows = cur.fetchall()
            conn.close()
            return '\n'.join(str(r) for r in rows)
        except Exception as e:
            conn.close()
            return f"Could not explain query: {e}"

    def summarize(self, nlq: str, result: Dict[str, Any]) -> str:
        # Use the model to summarize the SQL results for a non-technical user
        cols = result.get('columns', [])
        rows = result.get('rows', [])
        preview = [dict(zip(cols, r)) for r in rows[:5]]
        system = "You are a helpful assistant that summarizes SQL query results concisely."
        user = (
            f"User asked: {nlq}\nColumns: {cols}\nFirst rows (up to 5): {preview}\n"
            "Provide a short (2-4 sentence) summary and note how many rows were returned."
        )
        try:
            resp = chat([{"role":"system","content":system},{"role":"user","content":user}], temperature=0.2)
            return resp
        except Exception:
            return f"Returned {len(rows)} rows. Example row: {preview[:1]}"

    def handle(self, nlq: str, approved: bool = False) -> Dict[str, Any]:
        """Main flow: propose SQL, execute, explain plan, summarize."""
        proposal = self.suggest_sql(nlq)
        sql = proposal.get('sql','')
        explanation = proposal.get('explanation','')
        if not sql:
            return {"error":"no_sql_generated","proposal":proposal}
        exec_res = self.execute(sql, approved=approved)
        if exec_res.get("error") == "approval_required":
            return {
                "nlq": nlq,
                "sql": sql,
                "sql_explanation": explanation,
                "plan": "approval_blocked",
                "result": exec_res,
                "summary": "Execution blocked pending approval for high-risk query.",
            }
        plan = self.explain(sql)
        summary = self.summarize(nlq, exec_res)
        return {
            "nlq": nlq,
            "sql": sql,
            "sql_explanation": explanation,
            "plan": plan,
            "result": exec_res,
            "summary": summary,
        }
