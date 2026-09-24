"""Native experiment agents call this at the training boundary."""
import argparse
import json
from pathlib import Path
import sys


def feedback(benchmark, result):
    """Return task eligibility and the raw target; do not decide native KEEP."""
    sys.path.insert(0, benchmark.context['runner_root'])
    from engine.evaluation.objective import admissibility
    from engine.evaluation.task import Task
    raw = benchmark.task
    task = Task(raw['task_id'], raw['version'], raw['title'],
                benchmark.task_dir / 'task.json', raw)
    target = task.objective_spec['target']
    metrics = result.get('metrics', {})
    verdict = admissibility(task, metrics)
    value = metrics.get(target['metric'])
    eligible = result.get('status') == 'ok' and verdict.admissible and value is not None
    return {'metric_name': target['metric'], 'metric_value': value,
            'direction': target['direction'], 'eligible': eligible,
            'eligibility_reason': verdict.reason if result.get('status') == 'ok'
            else result.get('status', 'missing result status')}


def evaluate(benchmark, source, request_id, research_log, output, sentinel=None):
    """The caller retains its frozen source and ID; the API recovers that request."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    intent = {'source': str(Path(source).resolve()), 'request_id': request_id,
              'research_log': research_log}
    intent_path = output / 'request.json'
    if intent_path.exists():
        if json.loads(intent_path.read_text()) != intent:
            raise ValueError('Resume with the original source, request ID and research log')
    else:
        with intent_path.open('x') as stream:
            json.dump(intent, stream, indent=2)
    if sentinel is not None:
        sentinel = Path(sentinel)
        pending = json.loads(sentinel.read_text())
        pending['arena_request_dir'] = str(output.resolve())
        sentinel.write_text(json.dumps(pending, indent=2) + '\n')
        if pending.get('stdout_path'):
            with Path(pending['stdout_path']).open('a') as stream:
                stream.write(f'AutoArena request {request_id}: submitted or recovering\n')
    result = benchmark.evaluate(intent['source'], request_id, research_log=research_log)
    (output / 'arena_result.json').write_text(json.dumps(result, indent=2) + '\n')
    report = feedback(benchmark, result)
    report['arena_result'] = str(output / 'arena_result.json')
    (output / 'arena_feedback.json').write_text(json.dumps(report, indent=2) + '\n')
    if sentinel is not None:
        pending = json.loads(sentinel.read_text())
        pending.update(status='complete', val_score=report['metric_value'],
                       direction=report['direction'])
        sentinel.write_text(json.dumps(pending, indent=2) + '\n')
    return report


def recover_pending(pending_result):
    """Hydrate native resume feedback from the original canonical request."""
    from autoarena import Benchmark
    output = Path(pending_result['arena_request_dir'])
    intent = json.loads((output / 'request.json').read_text())
    return evaluate(Benchmark.from_context(), intent['source'], intent['request_id'],
                    intent['research_log'], output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--research-log', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sentinel', required=True, type=Path)
    args = parser.parse_args()
    from autoarena import Benchmark
    report = evaluate(Benchmark.from_context(), args.source, args.request_id,
                      json.loads(args.research_log.read_text()), args.output, args.sentinel)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
