import os, sys, json, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from tools import search, pdf_ingest, summarize


class ResearchExecutor:
    """Execute a research plan step-by-step using simple tools.

    This executor is intentionally small: each step is dispatched to a tool
    that returns structured output saved to `data/notes/`.
    """
    def __init__(self, data_dir=None):
        base = os.path.join(os.path.dirname(__file__), '..')
        self.data_dir = data_dir or os.path.abspath(os.path.join(base, 'data'))
        os.makedirs(os.path.join(self.data_dir, 'notes'), exist_ok=True)

    def _run_step_once(self, step: dict):
        action = step.get('action')
        if action == 'search':
            return search.search_web(step.get('description'))
        if action == 'ingest':
            return pdf_ingest.ingest_folder(os.path.join(self.data_dir, 'pdfs'))
        if action in ('extract', 'summarize'):
            return summarize.summarize_all(os.path.join(self.data_dir, 'pdfs'))
        if action == 'synthesize':
            return {'note': 'Synthesis placeholder: combine extracted notes into final brief.'}
        return {'note': f'No tool for action {action}'}

    def execute_plan(self, plan: dict) -> dict:
        qg = (plan.get('quality_gate') or {})
        if qg and not qg.get('valid', True):
            return {
                'plan_title': plan.get('title'),
                'error': 'plan_quality_gate_failed',
                'issues': qg.get('issues', []),
                'steps': [],
            }

        results = {'plan_title': plan.get('title'), 'steps': []}
        steps = sorted(plan.get('steps', []), key=lambda s: s.get('id', 10**9))
        completed = set()

        for step in steps:
            sid = step.get('id')
            action = step.get('action')
            depends_on = set(step.get('depends_on', []))
            if depends_on and not depends_on.issubset(completed):
                out = {'error': 'dependency_not_met', 'depends_on': sorted(depends_on), 'completed': sorted(completed)}
                status = 'skipped'
                latency_ms = 0
            else:
                status = 'ok'
                out = None
                started = time.perf_counter()
                retries = int(step.get('retries', 1))
                for attempt in range(1, retries + 1):
                    try:
                        out = self._run_step_once(step)
                        break
                    except Exception as e:
                        out = {'error': 'step_failed', 'attempt': attempt, 'message': str(e)}
                        if attempt == retries:
                            status = 'failed'
                latency_ms = int((time.perf_counter() - started) * 1000)

            fname = os.path.join(self.data_dir, 'notes', f'step_{sid}_{int(time.time())}.json')
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump({'step': step, 'status': status, 'latency_ms': latency_ms, 'result': out}, f, ensure_ascii=False, indent=2)

            if status == 'ok':
                completed.add(sid)
            results['steps'].append({
                'id': sid,
                'action': action,
                'status': status,
                'latency_ms': latency_ms,
                'result_summary': str(out)[:220],
            })
        return results
