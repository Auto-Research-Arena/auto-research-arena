"""Translate one canonical result into GEAR's controller score slots."""
import argparse
import json
import math
import sys
from pathlib import Path

from autoarena import objective_spec


def ranking_key(task):
    objective = objective_spec(task['objective'])
    target = objective['target']
    if target['direction'] != 'minimize':
        raise RuntimeError('The native GEAR scalar assumes minimization')
    return target['metric']


def measured(value):
    return type(value) in (int, float) and math.isfinite(value)


def target_divisor(key, reference):
    """Use this run's positive, finite reference measurement."""
    value = ((reference or {}).get('metrics') or {}).get(key)
    if not measured(value) or value <= 0:
        raise RuntimeError(f'This run requires a positive finite reference measurement for {key}')
    return value


def declares(task, metric):
    """Whether this task actually observes a metric, per its own frozen definition."""
    objective = task.raw['objective']
    names = set(task.raw.get('report', {}).get('columns') or ())
    names.add(objective['target']['metric'])
    if objective.get('quality_gate'):
        names.add(objective['quality_gate']['metric'])
    names.update(constraint['metric'] for constraint in objective.get('constraints', ()))
    return metric in names


def projection(result, key, settings, task, reference=None):
    from engine.evaluation.objective import admissibility

    if result['status'] in {'substrate_failure', 'infrastructure_failure', 'never_executed', 'preempted'}:
        raise RuntimeError('Infrastructure failure needs inspection before native feedback')
    metrics = result.get('metrics', {})
    # Require declared controller measurements; use 0.0 for unobserved slots.
    slots = {'memory_gb': ('peak_vram_bytes', settings['memory_divisor']),
             'params_m': ('num_params_total', settings['parameters_divisor'])}
    required = [key] + [metric for metric, _ in slots.values() if declares(task, metric)]
    if result['status'] != 'ok' or any(not measured(metrics.get(name)) for name in required):
        return {'status': 'crash'}
    divisor = target_divisor(key, reference)
    verdict = admissibility(task, metrics)
    projected = {slot: metrics[metric] / scale if measured(metrics.get(metric)) else 0.0
                 for slot, (metric, scale) in slots.items()}
    return {'status': 'ok', 'eligible': verdict.admissible,
            'eligibility_reason': verdict.reason, 'val_bpb': metrics[key] / divisor, **projected}



def main():
    from autoarena import Benchmark
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, help="Canonical result JSON; omit for this run's reference")
    args = parser.parse_args()
    benchmark = Benchmark.from_context()
    sys.path.insert(0, benchmark.context["runner_root"])
    from engine.evaluation.task import Task
    raw_task = benchmark.task
    task = Task(raw_task["task_id"], raw_task["version"], raw_task["title"],
                benchmark.task_dir / "task.json", raw_task)
    reference = benchmark.reference_result
    result = json.loads(args.result.read_text()) if args.result else reference
    settings = json.loads(Path(__file__).with_name("settings.json").read_text())
    print(json.dumps({"candidate_id": result["candidate_id"], "raw_result": result,
                      "controller_fields": projection(result, ranking_key(raw_task), settings, task, reference)}))


if __name__ == "__main__":
    main()
