"""Explain what is checked before a local benchmark run."""
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import uuid

from engine import ROOT
from engine.runtime import lifecycle as runtime
from engine.compute import load_backend
from engine.compute.storage import cache_environment, mount_command
from engine.evaluation.task import load_task


def probe(command, *, cwd=ROOT, env=None, timeout=120):
    """Run a small readiness command and clean up its own process group."""
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise ValueError('readiness command timed out')
    if process.returncode:
        raise ValueError(f'readiness command exited {process.returncode}: {(stderr or stdout)[-500:].strip()}')
    return stdout.strip()


def check_config(location):
    rows = []
    def check(name, action):
        try:
            detail = action()
            rows.append({'name': name, 'status': 'ready', 'detail': detail})
        except Exception as error:
            rows.append({'name': name, 'status': 'failed', 'detail': str(error)})
    try:
        config = runtime.load_config(location)
        method = runtime.interface(runtime.read(Path(config['method'])))
        task = load_task(config['task'])
        backend = load_backend(config['compute'])
    except Exception as error:
        return {'passed': False, 'checks': [{'name': 'Launch configuration', 'status': 'failed', 'detail': str(error)}]}
    rows.append({'name': 'Task definition', 'status': 'ready', 'detail': f'{task.task_id}: frozen objective, constraints and {task.launch["train_seconds"]}s training clock'})
    check('API interface', lambda: probe([sys.executable, '-c', 'from engine.autoarena import Benchmark; from engine.__main__ import main; print("Python client and engine CLI import successfully")']))
    def method_check():
        session_id = str(uuid.uuid4())
        run_dir, source, environment_root = runtime.launch_paths(config, session_id)
        if not source.is_dir():
            raise ValueError(f'method source is missing: {source}')
        runtime._launch_prompt(method, source)
        definition = {'config': config, 'method': method, 'api_version': 1,
                      'session_id': session_id,
                      'environment': {'root': str(environment_root)}}
        runtime.validate_commands(method, runtime.values(definition, run_dir))
        return f'{method["id"]}: start, resume and environment recipe valid; installs at launch'
    check('Method interface', method_check)
    workers, per_evaluation = method.get('workers', 1), task.launch['gpus_per_launch']
    devices = None
    def gpu_check():
        nonlocal devices
        devices = runtime.check_compute(backend, task, workers)
        return f'local devices {", ".join(devices)}; {workers} workers × {per_evaluation} GPUs per evaluation'
    check('Local GPUs', gpu_check)
    env = {**os.environ, **cache_environment(), **config['compute'].get('env', {})}
    env['CUDA_VISIBLE_DEVICES'] = ",".join(devices or [])
    mounts = config['compute'].get('mounts', {})
    def data_check():
        if mounts and not shutil.which('bwrap'):
            raise ValueError('dataset mounts require bubblewrap (bwrap)')
        for name, value in mounts.items():
            folder = Path(value)
            if not folder.is_dir() or not any(folder.iterdir()):
                raise ValueError(f'prepare the dataset directory: {folder}')
        return 'prepared data directories exist' if mounts else 'task uses its configured cache'
    check('Dataset', data_check)
    measurement = env.get('UV_PROJECT_ENVIRONMENT')
    if measurement and devices is not None:
        code = 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; a=torch.ones((8,8),device="cuda"); assert float((a@a).sum())==512; print("PyTorch " + torch.__version__ + "; CUDA operation passed on " + torch.cuda.get_device_name(0))'
        check('Measurement runtime', lambda: probe(mount_command([str(Path(measurement)/'bin/python'), '-c', code], mounts), env=env))
    elif measurement:
        rows.append({'name': 'Measurement runtime', 'status': 'skipped', 'detail': 'selected GPUs failed readiness'})
    else:
        rows.append({'name': 'Measurement runtime', 'status': 'skipped', 'detail': 'no measurement environment selected in compute.env'})
    llm = config.get('research_llm') or {}
    if llm.get('used') is False:
        rows.append({'name': 'LLM', 'status': 'skipped', 'detail': 'this method does not use an LLM'})
    elif llm.get('check'):
        def llm_check():
            command = llm['check']
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                raise ValueError('research_llm.check must be a command argument list')
            probe(command, env={**env, 'CUDA_VISIBLE_DEVICES': ''})
            return f'{llm.get("provider", "configured provider")} / {llm.get("model", "configured model")}: probe responded'
        check('LLM', llm_check)
    else:
        rows.append({'name': 'LLM', 'status': 'skipped', 'detail': 'no LLM probe requested in launch configuration'})
    return {'passed': all(row['status'] != 'failed' for row in rows), 'checks': rows}


def render(result):
    labels = {'ready': 'READY', 'failed': 'FAILED', 'skipped': 'SKIP'}
    lines = ['AutoArena readiness']
    for row in result['checks']:
        lines.append(f"  {labels[row['status']]:6}  {row['name']}: {row['detail']}")
    lines.append('Checks passed.' if result['passed'] else 'Fix the failed checks before launching.')
    return '\n'.join(lines)
