"""Package one recorded run as an offline submission with source and evidence."""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
import zipfile

from engine.records import runs
from engine.evaluation.task import Task
from engine.runtime import lifecycle as runtime
from engine.submission import record as submission, report as submission_report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')



def export(run_dir, output):
    """Create an offline record with exact source/log bytes; leave the run unchanged."""
    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise submission.SubmissionError(f'output already exists: {output}; choose a new directory')
    if run_dir == output or run_dir.is_relative_to(output):
        raise submission.SubmissionError('output must not contain the source run')
    record = submission.build(run_dir)
    if record['reconciliation']['disagreements']:
        raise submission.SubmissionError('submission does not reconcile with its launch ledger')
    run = runs.Run(run_dir, record["identity"])
    definition = record['task_definition']
    task = Task(definition["task_id"], definition["version"], definition["title"],
                run_dir / "task.json", definition)
    output.mkdir(parents=True)
    winner = record['headline']['best_candidate_id']
    reference = next((r['candidate_id'] for r in record['history'] if r['is_reference']), None)
    manifest, sources = {}, {}
    for role, candidate in (('reference', reference), ('best', winner)):
        if not candidate:
            continue
        identity = next(row["source_identity"] for row in record["history"]
                        if row["candidate_id"] == candidate)
        source = runs.candidate_source(run, candidate)
        names = sorted(set(task.substrate['mutable'] + task.substrate['immutable']))
        if not identity or not all((source / name).is_file() for name in names):
            continue
        if runtime.program_identity(task, source) != identity:
            raise submission.SubmissionError(f'archived source differs from recorded identity: {candidate}')
        manifest[role] = identity
        sources[role] = []
        for name in names:
            dest = output / 'code' / role / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = (source / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != identity['files'][name]:
                raise submission.SubmissionError(f'source changed during export: {candidate}/{name}')
            dest.write_bytes(data)
            sources[role].append({'name': name, 'path': str(dest.relative_to(output)), 'bytes': len(data)})
    write_json(output / 'evidence/source-manifest.json', manifest)
    changes, changed = [], []
    if 'best' in sources and 'reference' in sources:
        for item in sources['best']:
            name = item['name']
            before = (output / 'code/reference' / name).read_bytes()
            after = (output / 'code/best' / name).read_bytes()
            if before != after:
                changed.append(name)
                text_diff = list(difflib.unified_diff(
                    before.decode('utf-8', errors='replace').splitlines(True),
                    after.decode('utf-8', errors='replace').splitlines(True),
                    fromfile='reference/' + name, tofile='best/' + name))
                changes.extend(text_diff or [f'Byte contents differ: reference/{name} and best/{name}\n'])
        (output / 'code/changes.diff').write_text(''.join(changes))
    write_json(output / 'reproduce/task.json', definition)
    write_json(output / 'reproduce/evaluation.json', {
        'entrypoint': definition['substrate']['entrypoint'], 'launch': definition['launch'],
        'source': '../code/best', 'source_identity': manifest.get('best'),
        'benchmark_revision': run.meta.get('harness_revision'), 'setup': record['setup']})
    (output / 'reproduce/README.md').write_text(
        '# Reproduce this evaluation\n\n'
        'The measured source and dependency files are in `../code/best/`. '
        '`evaluation.json` records the entrypoint, launch environment, GPU requirement and timeout.\n\n'
        '1. Prepare the data and measurement environment described in the benchmark task guide.\n'
        '2. Copy `code/best/` to local evaluation scratch and use its pinned dependencies.\n'
        '3. Run the recorded entrypoint with the environment and timeout in `evaluation.json`.\n'
        '4. Compare the instrument output using the frozen objective and constraints in `task.json`.\n\n'
        'Use the benchmark evaluator at the recorded revision for a canonical measurement.\n')
    record['winning_implementation'] = {'candidate_id': winner, 'source_verified': 'best' in sources,
        'files': sources.get('best', []), 'changed_files': changed}
    if 'best' in sources:
        with zipfile.ZipFile(output / 'code/best.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for folder in ('code/best', 'code/reference', 'reproduce'):
                for path in sorted((output / folder).rglob('*')):
                    if path.is_file():
                        archive.write(path, str(path.relative_to(output)))
            for name in ('code/changes.diff', 'evidence/source-manifest.json'):
                if (output / name).is_file():
                    archive.write(output / name, name)
        record['winning_implementation'].update(source='code/best/', download='code/best.zip', reproduce='reproduce/README.md')
        if 'reference' in sources:
            record['winning_implementation'].update(reference='code/reference/', diff='code/changes.diff')
    if winner and ('best' not in sources or 'reference' not in sources):
        record['verdict']['publishable'] = False
        record['verdict']['reasons'].append('Exact winning and reference source are required for a complete submission.')
    logs = {}
    for row in record['history']:
        for channel in ('stdout', 'stderr'):
            relative = row.get(channel)
            source = runs.log_path(run, relative) if relative else None
            if source is None:
                found = list((run_dir / 'logs' / f'{row["launch_seq"]:04d}').glob(f'*/{channel}.log'))
                if len(found) == 1:
                    source = found[0]
                    row[channel + '_basis'] = 'Unique surviving file in launch directory; ledger path absent.'
            row[channel] = None
            if source is None or not source.is_file():
                continue
            relative = str(source.relative_to(run_dir))
            dest = output / 'evidence' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw = source.read_bytes()
            dest.write_bytes(raw)
            logs[relative] = {'path': str(dest.relative_to(output)),
                              'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            row[channel] = logs[relative]['path']
        row['research_account'] = f'evidence/ideas/launch-{row["launch_seq"]:04d}.html'
        role = 'best' if row['candidate_id'] == winner else 'reference' if row['candidate_id'] == reference else None
        row['source']['included_in_submission'] = role in sources
        row['source']['path'] = f'code/{role}/' if role in sources else None
    for entry in record['evidence']:
        row = next(r for r in record['history'] if r['launch_seq'] == entry['launch_seq'])
        for channel in ('stdout', 'stderr'):
            entry[channel] = row[channel]
            entry[channel + '_bytes'] = (output / row[channel]).stat().st_size if row[channel] else None
    for row in record['history']:
        path = output / row['research_account']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(submission_report.render_account(record, row))
    for name, data in (('log-manifest', logs), ('history', record['history']), ('audit', record['audit']), ('reconciliation', record['reconciliation'])):
        write_json(output / f'evidence/{name}.json', data)
    write_json(output / 'submission.json', record)
    (output / 'submission.html').write_text(submission_report.build(record))
    return record
