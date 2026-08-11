import sys
import os
from agents.planner import plan_research
from agents.executor import ResearchExecutor

def main(prompt: str):
    # Ensure data folders
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(os.path.join(data_dir, 'pdfs'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'notes'), exist_ok=True)

    print('Generating research plan...')
    p = plan_research(prompt)
    print('\n== Plan ==\n', p)

    print('\n== Executing plan ==')
    execer = ResearchExecutor()
    result = execer.execute_plan(p)
    print('\n== Execution Result ==\n', result)

if __name__ == '__main__':
    q = ' '.join(sys.argv[1:]) or 'Survey methods for low-resource named-entity recognition'
    main(q)
