import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from capstones.capstone1_sql_agent.agent import SQLiteAnalystAgent
from capstones.capstone1_sql_agent.data.setup_db import init_database

def ensure_db():
    """Ensure database exists and is initialized."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    db_path = os.path.join(data_dir, 'sample.db')
    
    if not os.path.exists(db_path):
        print("Initializing database...")
        # Change to data directory to find employees.csv
        original_dir = os.getcwd()
        os.chdir(data_dir)
        try:
            init_database(db_path)
        finally:
            os.chdir(original_dir)
    
    return db_path


def main(nlq: str):
    db = ensure_db()
    agent = SQLiteAnalystAgent(db)
    out = agent.handle(nlq)
    # print readable output for students
    print('Natural language:', out.get('nlq'))
    print('\nGenerated SQL:\n', out.get('sql'))
    print('\nSQL Explanation:\n', out.get('sql_explanation'))
    print('\nQuery Plan:\n', out.get('plan'))
    print('\nResult preview (columns + up to 5 rows):')
    res = out.get('result', {})
    print(res.get('columns'))
    for r in res.get('rows', [])[:5]:
        print(r)
    print('\nSummary:\n', out.get('summary'))


if __name__ == '__main__':
    nlq = ' '.join(sys.argv[1:]) or 'List engineering employees with salary > 2,000,000'
    main(nlq)