"""Initialize a sample SQLite database with richer schema and sample data.

This script creates multiple related tables to support more realistic
queries (joins, aggregates, many-to-many relationships):
- departments
- employees
- projects
- employee_projects
- salary_history

If a lightweight `employees.csv` exists in the same folder the script will
import it into `employees`. Otherwise the script inserts built-in sample data.
"""

import sqlite3
import csv
import os
from datetime import date

DB = 'sample.db'


def _execute(conn, sql, params=None):
    cur = conn.cursor()
    cur.execute(sql) if params is None else cur.execute(sql, params)
    return cur


def init_database(db_path=DB):
    """Create schema and populate with sample data.

    Args:
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)

    # Create tables
    _execute(conn, """
    CREATE TABLE IF NOT EXISTS departments (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE
    )
    """)

    _execute(conn, """
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        dept_id INTEGER REFERENCES departments(id),
        manager_id INTEGER REFERENCES employees(id),
        hire_date DATE,
        current_salary INTEGER
    )
    """)

    _execute(conn, """
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY,
        name TEXT,
        start_date DATE
    )
    """)

    _execute(conn, """
    CREATE TABLE IF NOT EXISTS employee_projects (
        employee_id INTEGER REFERENCES employees(id),
        project_id INTEGER REFERENCES projects(id),
        role TEXT,
        PRIMARY KEY (employee_id, project_id)
    )
    """)

    _execute(conn, """
    CREATE TABLE IF NOT EXISTS salary_history (
        id INTEGER PRIMARY KEY,
        employee_id INTEGER REFERENCES employees(id),
        salary INTEGER,
        start_date DATE
    )
    """)

    # Clear existing sample rows (safe for re-initialization)
    for t in ("employee_projects", "salary_history", "employees", "projects", "departments"):
        _execute(conn, f"DELETE FROM {t}")

    # Prefer CSV import if available (backwards compatible)
    csv_path = 'employees.csv'
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # collect departments
            depts = {}
            for row in reader:
                dept = row.get('dept', 'Unknown')
                depts.setdefault(dept, []).append(row)
            # insert departments
            for name in depts:
                _execute(conn, 'INSERT INTO departments(name) VALUES(?)', (name,))
            conn.commit()
            # insert employees from CSV
            for name in depts:
                # get dept id
                cur = _execute(conn, 'SELECT id FROM departments WHERE name=?', (name,))
                dept_id = cur.fetchone()[0]
                for r in depts[name]:
                    _execute(conn, 'INSERT INTO employees(name, dept_id, hire_date, current_salary) VALUES (?, ?, ?, ?)',
                             (r['name'], dept_id, date.today().isoformat(), int(r['salary'])))

    else:
        # Insert sample departments
        depts = ['Engineering', 'HR', 'Sales', 'Finance']
        for d in depts:
            _execute(conn, 'INSERT INTO departments(name) VALUES(?)', (d,))

        # Insert sample employees with managers and hire dates
        employees = [
            ('Asha', 'Engineering', None, '2019-03-01', 2200000),
            ('Ravi', 'Engineering', 'Asha', '2020-06-15', 1800000),
            ('Tanvi', 'Engineering', 'Asha', '2018-11-11', 2600000),
            ('Neha', 'HR', None, '2017-01-08', 1200000),
            ('Ishaan', 'Sales', None, '2021-09-20', 1500000),
        ]

        # map dept name to id
        dept_map = {row[0]: row[0] for row in []}
        cur = _execute(conn, 'SELECT id, name FROM departments')
        dept_map = {name: id for id, name in cur.fetchall()}

        # insert employees without manager_id first
        for name, dept, manager, hire, sal in employees:
            dept_id = dept_map.get(dept)
            _execute(conn, 'INSERT INTO employees(name, dept_id, hire_date, current_salary) VALUES (?, ?, ?, ?)',
                     (name, dept_id, hire, sal))

        # set manager relationships (by name -> id)
        conn.commit()
        cur = _execute(conn, 'SELECT id, name FROM employees')
        emp_map = {name: id for id, name in cur.fetchall()}
        # simple manager assignments
        mgr_pairs = [('Ravi', 'Asha'), ('Tanvi', 'Asha')]
        for emp, mgr in mgr_pairs:
            emp_id = emp_map.get(emp)
            mgr_id = emp_map.get(mgr)
            if emp_id and mgr_id:
                _execute(conn, 'UPDATE employees SET manager_id=? WHERE id=?', (mgr_id, emp_id))

        # Insert projects and assignments
        projects = [('Apollo', '2022-01-01'), ('Hermes', '2023-06-01')]
        for name, start in projects:
            _execute(conn, 'INSERT INTO projects(name, start_date) VALUES (?, ?)', (name, start))
        conn.commit()
        cur = _execute(conn, 'SELECT id, name FROM projects')
        proj_map = {name: id for id, name in cur.fetchall()}

        # Assign employees to projects
        assignments = [('Asha', 'Apollo', 'Lead'), ('Ravi', 'Apollo', 'Contributor'), ('Ishaan', 'Hermes', 'Sales')]
        for en, pn, role in assignments:
            e_id = emp_map.get(en)
            p_id = proj_map.get(pn)
            if e_id and p_id:
                _execute(conn, 'INSERT INTO employee_projects(employee_id, project_id, role) VALUES (?, ?, ?)',
                         (e_id, p_id, role))

        # salary history
        for name in emp_map:
            _execute(conn, 'INSERT INTO salary_history(employee_id, salary, start_date) VALUES (?, ?, ?)',
                     (emp_map[name], 1000000 + 100000 * (emp_map[name] % 5), '2020-01-01'))

    conn.commit()
    conn.close()
    print(f'✓ Database initialized: {db_path}')


if __name__ == '__main__':
    init_database()
