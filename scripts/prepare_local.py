#!/usr/bin/env python3
"""Install the frozen measurement dependencies and prepare repository-local data."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine import ROOT
from engine.compute.storage import cache_environment, mount_command
from engine.evaluation.task import load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='params')
    args = parser.parse_args()
    task = load_task(args.task)
    uv = shutil.which('uv')
    if not uv:
        parser.error('install uv first')
    env = {**os.environ, **cache_environment(), 'UV_PROJECT_ENVIRONMENT': str(ROOT / '.local/measurement')}
    for key, path in cache_environment().items():
        if key != "UV_LINK_MODE":
            Path(path).mkdir(parents=True, exist_ok=True)
    env.pop('UV_NO_SYNC', None)
    env.pop('UV_NO_PROJECT', None)
    data = ROOT / 'data/autoresearch'
    data.mkdir(parents=True, exist_ok=True)
    mounts = {'~/.cache/autoresearch': str(data)}
    subprocess.run([uv, 'sync', '--project', str(task.source_root), '--frozen', '--python', '3.10'], env=env, check=True)
    subprocess.run(mount_command(task.substrate['prepare'], mounts), cwd=task.source_root, env=env, check=True)
    print('Measurement environment: .local/measurement\nDataset and tokenizer: data/autoresearch')


if __name__ == '__main__':
    main()
