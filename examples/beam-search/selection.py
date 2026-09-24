"""Read-only objective arithmetic for Beam Search roles.

The lightweight client returns raw measurements. This helper reads the frozen
task JSON and computes eligibility, program identity and gated ranking values.
The evaluator owns retune collapse, exact ties and frontier selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from autoarena import objective_spec

OPERATORS = {'<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
             '>': lambda a, b: a > b, '>=': lambda a, b: a >= b,
             '==': lambda a, b: a == b}


def ranking_metrics(task):
    objective = objective_spec(task['objective'])
    if objective['comparison_mode'] != 'gated':
        raise ValueError('Beam Search supports gated task definitions only')
    return [objective['target'], *objective['tiebreaks']]


def gate_margin(task, metrics):
    gate = objective_spec(task['objective']).get('quality_gate')
    if gate is None:
        return None
    value = metrics.get(gate['metric'])
    if value is None:
        return None
    if gate['operator'] in ('<', '<='):
        return float(gate['value']) - float(value)
    return float(value) - float(gate['value'])


def arithmetic(task, metrics):
    objective = objective_spec(task['objective'])
    entries = ranking_metrics(task)
    for index, limit in enumerate(objective.get('constraints', [])):
        value = metrics.get(limit['metric'])
        clause = f"ceiling[{index}]:{limit['metric']}"
        if value is None:
            return {'admissible': False, 'clause': clause + ':unmeasured', 'rank_key': None}
        if not OPERATORS[limit['operator']](float(value), float(limit['value'])):
            return {'admissible': False, 'clause': clause, 'rank_key': None}
    gate = objective.get('quality_gate')
    if gate is not None:
        value = metrics.get(gate['metric'])
        clause = 'gate:' + gate['metric']
        if value is None:
            return {'admissible': False, 'clause': clause + ':unmeasured', 'rank_key': None}
        if not OPERATORS[gate['operator']](float(value), float(gate['value'])):
            return {'admissible': False, 'clause': clause, 'rank_key': None}
    for entry in entries:
        value = metrics.get(entry['metric'])
        if value is None:
            continue
        if not math.isfinite(float(value)):
            return {'admissible': False, 'clause': 'validity_floor:' + entry['metric'] + ':non_finite', 'rank_key': None}
        if float(value) <= 0:
            return {'admissible': False, 'clause': 'validity_floor:' + entry['metric'] + ':non_positive', 'rank_key': None}
    margin = gate_margin(task, metrics)
    if any(metrics.get(entry['metric']) is None for entry in entries):
        return {'admissible': True, 'clause': 'admissible', 'rank_key': None, 'gate_margin': margin}
    key = []
    for entry in entries:
        if entry['direction'] not in ('minimize', 'maximize'):
            raise ValueError('Unknown ranking direction')
        value = float(metrics[entry['metric']])
        key.append(value if entry['direction'] == 'minimize' else -value)
    if margin is not None:
        key.append(-margin)
    return {'admissible': True, 'clause': 'admissible', 'rank_key': key, 'gate_margin': margin}


def program_identity(task, source):
    """Same exact declared-file identity as engine.runtime.lifecycle.program_identity."""
    root = Path(source)
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in sorted(set(task['substrate']['mutable'] + task['substrate']['immutable']))}
    encoded = (json.dumps({'task': task, 'files': files}, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    return {'files': files, 'sha256': hashlib.sha256(encoded).hexdigest()}


def inspect_pool(task, entries):
    """Describe every entry; native evaluator performs beam-specific decisions."""
    ids = [entry['id'] for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError('Pool candidate IDs must be unique')
    result, identities = [], {}
    for entry in entries:
        if entry.get('experimental_variable') is not None:
            raise ValueError('Current task API has no external experimental-variable input')
        identity = program_identity(task, entry['source'])
        verdict = arithmetic(task, entry['metrics'])
        value = {'id': entry['id'], 'parent_id': entry.get('parent_id'),
                 'source_identity': identity, 'execution_status': entry['status'],
                 'objective': verdict}
        if entry['status'] != 'ok':
            value['excluded'] = 'execution_not_ok'
        elif identity['sha256'] in identities:
            value['excluded'] = 'duplicate_program'
            value['duplicate_of'] = identities[identity['sha256']]
        elif not verdict['admissible'] or verdict['rank_key'] is None:
            value['excluded'] = verdict['clause'] if not verdict['admissible'] else 'ranking_unmeasured'
        else:
            identities[identity['sha256']] = entry['id']
        if 'excluded' in value:
            verdict['rank_key'] = None
        result.append(value)
    return {'entries': result, 'selection_owner': 'native evaluator',
            'note': 'Apply measured-neutral parent/child collapse only when the retune lane is open this generation. Apply exact-tie fallback before publishing the beam.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--pool', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    task = json.loads(args.task.read_text())
    pool = json.loads(args.pool.read_text())
    result = inspect_pool(task, pool)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
