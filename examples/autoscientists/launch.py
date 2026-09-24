"""Start the private native service and launcher, then hand research to Claude."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from records import export_records


def binding(benchmark):
    return f'''## AutoArena experiment binding

Read the complete task in `{benchmark.task_dir}/task.json`, including every
constraint, and this run's measured reference in `AUTOARENA_CONTEXT`. The supplied
reference code replaces downloading a training repository. Dependencies and data
are prepared by the benchmark. Two benchmark workers execute the experiments.

Keep the native teams, discussion, queues, claims, scheduling and periodic hooks.
Each GPU experiment agent owns its complete claim, build, evaluate, inspect,
record, release and publish workflow. Benchmark workers provide its GPU; the
agent does not need a visible local device or a local training environment.
Replace direct training (including any native confirmation evaluation) with:

```bash
{sys.executable} {ROOT / 'evaluate.py'} --source FROZEN_SOURCE --request-id ID --research-log RESEARCH_LOG_JSON --output RESULT_DIRECTORY --sentinel NATIVE_RESULT_LATEST_JSON
```

Before submission, preserve each complete candidate in a separate
`{benchmark.workspace}/candidates/ID` directory. Keep that source unchanged for
request recovery. The research-log JSON contains a nonempty ideas list and a
nonblank status, including the actual proposal and current native progress.
Create the native Step 4 in-flight sentinel first. The helper adds its durable
arena_request_dir to that sentinel before evaluation. Preserve that field in
subsequent native sentinel updates, including multi-seed request records.
The helper persists the request identity, writes the unmodified canonical result
to arena_result.json, and supplies arena_feedback.json. Read that feedback into
`arena_feedback`; use its raw `metric_value` as `our_metric`, its metric name and
direction, and inspect every measurement, log and constraint failure in the full
result. Never substitute val_bpb for the declared target. Retain native strict
comparison against the fresh champion; ineligible results cannot be KEEP or
promoted, and their measurements and insights still enter native history.
The GPU agent publishes its result immediately through the native files and
posts. It also owns champion publication under ROLE-GPU Step 7b/7b1 with the
existing version checks and atomic copy; the orchestrator observes completion
and generates native follow-ups, without promoting the same code again.
Recover interrupted API calls using their original request IDs and frozen inputs.
The reference is already measured and must not be rerun.

Keep native multi-seed selection and confirmation logic. Every selected seed
must obey the frozen task contract. If it requires a different seed that the task
forbids, record the requested seed and exact conflicting task clause as blocked
confirmation; leave the candidate unconfirmed and do not promote it. Do not
replace a second seed with the first result or silently waive confirmation.
Record full canonical result paths alongside native experiment/champion records.

Credentials remain in private service files; never print or copy them into research
logs or the benchmark run directory.

'''


# The run binding above is prepended to every agent's heartbeat as well as to the
# orchestrator prompt, so its `autoarena finish` sentence reaches agents that must
# never make that call. Saying so once here is cheaper than every agent reasoning
# its way to the same refusal on every heartbeat.
AGENT_SCOPE = '''
`autoarena finish` above is the orchestrator's single call at the end of the run.
You are an analyst, experiment or monitor agent: you never call it, and you do not
need to weigh whether to. Complete your own role's step and exit.

'''


def champion_record(benchmark):
    """Initialize the reproduction record written by native ROLE-GPU Step 7b."""
    import yaml
    target = benchmark.task['objective']['target']
    value = benchmark.reference_result['metrics'][target['metric']]
    fields = {'metric_name': target['metric'], 'metric_value': value,
              'seed_values': [value], 'direction': target['direction'],
              'experiment_id': 'reference', 'agent': 'benchmark'}
    return ('---\n' + yaml.safe_dump(fields, sort_keys=False) + '---\n\n'
            '# Champion: measured reference\n\n'
            'Source: `champion/train.py`. Full canonical measurement: '
            '`champion/arena_result.json`.\n\n'
            'The reproduction recipe and constraints are in `task/TASK.md`; '
            'the complete supplied source is in `champion/`.\n')


def task_profile(text, benchmark):
    """Resolve promotion ownership and bind only the big-win trigger's units."""
    start = text.index('**Champion =')
    end = text.index('### Auto-bracket big wins', start)
    text = text[:start] + (
        'The KEEP-winning GPU agent publishes champion.md and champion source '
        'using ROLE-GPU Steps 7b and 7b1, including version checks and atomic copy. '
        'The orchestrator observes and logs that completed publication; it does '
        'not copy or promote the same candidate again.\n\n') + text[end:]
    target = benchmark.task['objective']['target']['metric']
    reference = benchmark.reference_result['metrics'][target]
    if reference <= 0:
        raise ValueError('Auto-bracketing requires a positive reference target')
    text = text.replace('abs(delta) > 0.001', 'abs(delta) / reference_target > 0.001')
    text = text.replace('### Auto-bracket big wins\n',
        '### Auto-bracket big wins\n\n'
        f'For this run, `reference_target = {reference!r}` ({target}). '
        '`delta` remains the raw target difference. Only this follow-up trigger '
        'uses reference normalization: 0.001 means 0.1% of the measured reference.\n')
    return text.replace('```python\nif outcome == "KEEP" and abs(delta)',
                        f'```python\nreference_target = {reference!r}\n'
                        'if outcome == "KEEP" and abs(delta)')


def adapt_role(text):
    """Bind native feedback, preserving research, noise gates and posting flow."""
    start = text.find('### Step 1 — Check GPU Availability')
    if start >= 0:
        end = text.index('### Step 1.5', start)
        text = text[:start] + ('### Step 1 — Benchmark GPU Availability\n\n'
            'The benchmark provides measurement workers through evaluate.py. '
            'Continue the native claim/build/evaluate workflow using those workers; '
            'the experiment-agent process has no local GPU assignment.\n\n') + text[end:]
    text = text.replace(
        'current_best = fresh_champ.get(metric_name, float("inf") if direction == "minimize" else float("-inf"))',
        'current_best = fresh_champ["metric_value"]')
    text = text.replace(
        'current_best = champ.get(metric_name, float("-inf") if direction == "maximize" else float("inf"))',
        'current_best = champ["metric_value"]')
    text = text.replace('elif (direction == "minimize" and our_metric < current_best)',
                        'elif not arena_feedback["eligible"]:\n    outcome = "DISCARD"\n'
                        'elif (direction == "minimize" and our_metric < current_best)')
    start = text.find('**Pattern A (default, foreground shell).**')
    if start >= 0:
        end = text.index('### Step 4b', start)
        text = text[:start] + '''Evaluate through the benchmark in this same experiment-agent session.
Use a distinct native experiment ID for each requested confirmation seed, preserving
native multi-seed checks. The seed must be permitted by the frozen task.

```python
import json, os, shutil
from pathlib import Path
from autoarena import Benchmark
from evaluate import evaluate

benchmark = Benchmark.from_context()
ws = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace")
rep = ws / "repo"
request_id = f"{AGENT_NAME}-{exp_id}"
frozen = benchmark.workspace / "candidates" / request_id
if not frozen.exists():
    shutil.copytree(rep, frozen)
sentinel_path = ws / "result_latest.json"
sentinel = {
    "status": "running", "posted_to_workshop": False,
    "exp_id": exp_id, "agent": AGENT_NAME, "item": item,
    "queue_claimed": True, "direction": direction, "val_score": None,
    "train_path": str(rep / f"train_{exp_id}.py"),
    "stdout_path": str(ws / f"train_{exp_id}.stdout"),
    "pid": os.getpid(), "description": description,
}
sentinel_path.write_text(json.dumps(sentinel, indent=2))
research_log = {"ideas": [{"description": description, "proposal": item}],
                "status": f"Evaluating native claimed experiment {exp_id}"}
arena_feedback = evaluate(benchmark, frozen, request_id, research_log,
                          ws / "arena" / exp_id, sentinel_path)
our_metric = arena_feedback["metric_value"]
direction = arena_feedback["direction"]
training_succeeded = json.loads(Path(arena_feedback["arena_result"]).read_text())["status"] == "ok"
```

Inspect the complete canonical result and logs, then continue Steps 4b–8.

''' + text[end:]
    start = text.find('our_metric  = pending_result.get("val_score")')
    if start >= 0:
        end = text.index('# 5b. KEEP/DISCARD/FAILED', start)
        text = text[:start] + '''from evaluate import recover_pending
arena_feedback = recover_pending(pending_result)
our_metric = arena_feedback["metric_value"]
direction = arena_feedback["direction"]
item = pending_result.get("item") or {}
description = pending_result.get("description") or item.get("diff") or f"Resumed ({exp_id})"

''' + text[end:]
        text = text.replace('improved = (direction == "maximize" and our_metric > current_best) or \\\n           (direction == "minimize" and our_metric < current_best)',
            'improved = arena_feedback["eligible"] and ((direction == "maximize" and our_metric > current_best) or \\\n           (direction == "minimize" and our_metric < current_best))')
        text = text.replace('if not diff_applied:\n    outcome = "FAILED"\nelse:',
                            'if not diff_applied or our_metric is None:\n    outcome = "FAILED"\nelse:')
        text = text.replace('delta   = (our_metric - current_best) if direction == "maximize" else (current_best - our_metric)',
                            'delta = 0.0 if our_metric is None else ((our_metric - current_best) if direction == "maximize" else (current_best - our_metric))')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--agent')
    parser.add_argument('--model')
    parser.add_argument('--session-id')
    args = parser.parse_args()
    from autoarena import Benchmark
    service = Path(sys.prefix) / 'node_modules/clawinstitute/bin/clawinstitute.js'
    native = ROOT / 'upstream'
    if not service.is_file() or not shutil.which('node') or not shutil.which('claude'):
        raise RuntimeError('Install ClawInstitute and provide node and Claude on PATH')
    node_version = subprocess.check_output(['node', '-p', 'process.versions.node'], text=True)
    if int(node_version.split('.')[0]) < 22:
        raise RuntimeError('ClawInstitute requires Node.js 22 or newer')
    for name in ('launch.py', 'runbook.md', 'task-autoresearch/LAUNCH.md'):
        if not (native / name).is_file():
            raise RuntimeError('Initialize the pinned AutoScientists source: ' + name)
    if args.check:
        print('Native AutoScientists launcher, ClawInstitute and Claude ready')
        return
    benchmark = Benchmark.from_context()
    settings = json.loads((ROOT / 'settings.json').read_text())
    key = hashlib.sha256(str(benchmark.run_dir).encode()).hexdigest()[:16]
    scratch = Path(os.environ['TMPDIR']) / 'autoscientists' / key
    scratch.mkdir(parents=True, exist_ok=True)
    private = scratch / 'service'
    private.mkdir(mode=0o700, exist_ok=True)
    token_file = private / 'token'
    if not token_file.exists():
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write('claw_' + secrets.token_hex(32))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    env = dict(os.environ)
    env.pop('DATABASE_URL', None)
    env.update(PORT=str(port), CLAWINSTITUTE_TOKEN=token_file.read_text().strip(),
               CLAWINSTITUTE_HOME=str(private), CLAWINSTITUTE_DB_DIR=str(private / 'db'),
               CLAWINSTITUTE_AUTH_REQUIRED='1', CLAWINSTITUTE_SKIP_FRONTEND='1', CUDA_VISIBLE_DEVICES='',
               WORKSHOP_NAME='aa_' + key)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    endpoint = f'http://127.0.0.1:{port}/api/v1'
    env['CLAWINSTITUTE_API'] = endpoint
    log_path = private / 'service.log'
    with log_path.open('ab') as log:
        child = subprocess.Popen(['node', str(service), 'start'], cwd=private, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    focus = scratch / 'native' / benchmark.run_dir.name
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        for _ in range(60):
            if child.poll() is not None:
                raise RuntimeError(f'Private ClawInstitute service exited during startup; inspect {log_path}')
            try:
                request = urllib.request.Request(endpoint + '/workshops', headers={'Authorization': 'Bearer ' + env['CLAWINSTITUTE_TOKEN']})
                with urllib.request.urlopen(request, timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError(f'Private ClawInstitute service did not become ready; inspect {log_path}')
        instructions = binding(benchmark)
        if not args.resume:
            task = native / ('.autoarena-task-' + key)
            task.mkdir(exist_ok=False)
            target = benchmark.task['objective']['target']
            (task / 'TASK.md').write_text(f'---\nname: {benchmark.task["task_id"]}\ntask_type: optimization\nmetric: {target["metric"]}\ndirection: {target["direction"]}\n---\n\n' + instructions + json.dumps(benchmark.task, indent=2))
            (task / 'LAUNCH.md').write_text(instructions + task_profile(
                (native / 'task-autoresearch/LAUNCH.md').read_text(), benchmark))
            shutil.copytree(benchmark.task_dir / 'code', task / 'repo')
            shutil.copytree(benchmark.task_dir / 'code', task / 'champion')
            reference = json.dumps(benchmark.reference_result, indent=2)
            (task / 'champion/arena_result.json').write_text(reference + '\n')
            (task / 'champion/SOURCE').write_text('AutoArena measured reference\n')
            launched = subprocess.run([sys.executable, str(native / 'launch.py'), benchmark.run_dir.name,
                                      '--task', str(task), '--output-dir', str(focus.parent)],
                                     env=env, cwd=scratch, capture_output=True, text=True)
            if launched.returncode:
                diagnostic = scratch / 'bootstrap.log'
                diagnostic.write_text(launched.stdout + '\n' + launched.stderr)
                raise RuntimeError(f'Native AutoScientists bootstrap failed; inspect {diagnostic}')
            champion = champion_record(benchmark)
            (focus / 'champion.md').write_text(champion)
            workspace_id = (focus / 'WORKSPACE_ID').read_text().strip()
            for name, content in {'champion.md': champion,
                                  'champion/train.py': (task / 'champion/train.py').read_text(),
                                  'champion/arena_result.json': reference}.items():
                request = urllib.request.Request(endpoint + f'/workspaces/{workspace_id}/files/{name}',
                    data=json.dumps({'content': content}).encode(), method='PUT',
                    headers={'Authorization': 'Bearer ' + env['CLAWINSTITUTE_TOKEN'], 'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=10) as response:
                    if response.status >= 300:
                        raise RuntimeError('Native shared champion initialization failed')
            # These are run-local task bindings; the pinned native runbook is unchanged.
            for path in focus.glob('agents/*/HEARTBEAT.md'):
                path.write_text(instructions + AGENT_SCOPE + adapt_role(path.read_text()))
            for name in ('ROLE-GPU.md', 'HEARTBEAT.md'):
                path = focus / 'system/templates' / name
                path.write_text(instructions + AGENT_SCOPE + adapt_role(path.read_text()))
        elif not (focus / 'WORKSPACE_ID').exists():
            raise RuntimeError('Native AutoScientists workspace is missing; cannot resume')
        (benchmark.workspace / 'native-state.json').write_text(json.dumps({'workspace': str(focus), 'service': endpoint}, indent=2) + '\n')
        prompt = instructions + f'\nNative workspace: {focus}. Read its task-profile.md and runbook.md and execute the native research loop.\n' + sys.stdin.read()
        command = [sys.executable, '-m', 'examples.claude_transport', '--', args.agent,
                   '--dangerously-skip-permissions', '--model', args.model,
                   '--resume' if args.resume else '--session-id', args.session_id]
        result = subprocess.run(command, cwd=benchmark.context['runner_root'], env=env, input=prompt, text=True)
        raise SystemExit(result.returncode)
    finally:
        if (focus / 'WORKSPACE_ID').is_file():
            destination = benchmark.workspace / 'native-record'
            try:
                export_records(focus, destination, endpoint,
                               env['CLAWINSTITUTE_TOKEN'], env['WORKSHOP_NAME'])
            except Exception as error:
                # Keep the private state available for recovery; never print an
                # API response or credentials in a shutdown diagnostic.
                destination.mkdir(parents=True, exist_ok=True)
                (destination / 'export-error.json').write_text(json.dumps({
                    'error_type': type(error).__name__,
                    'message': 'Native export incomplete; private workspace/service data retained for recovery',
                }) + '\n')
                print('Native record export incomplete; see native-record/export-error.json', file=sys.stderr)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
