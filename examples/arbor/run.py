"""Launch Arbor with native tree/merge defaults and an explicit cycle limit.

Tree depth (2) and merge threshold (5.0) come from ``CoordinatorConfig``;
settings raise the cycle limit from 40 to 100. The installer adds benchmark
measurement instructions to both native system prompts. See README.md for
the integration changes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from autoarena import Benchmark
from bridge import project, store_reference, target, task_object, write_json

ROOT = Path(__file__).resolve().parent


def configuration(benchmark, settings):
    code = benchmark.workspace / "arbor-code"
    state = benchmark.workspace / "arbor-native"
    command = (f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'bridge.py'))} "
               "--source . --research-log research_log.json")
    instructions = (ROOT / "prompt.md").read_text().replace("{eval_cmd}", command)
    task_text = (instructions + "\n\nFrozen task definition:\n"
                 + json.dumps(benchmark.task, indent=2)
                 + "\n\nCanonical reference result:\n"
                 + json.dumps(benchmark.reference_result, indent=2))
    config = {
        # Tree depth and merge threshold retain native defaults.
        "cwd": str(code), "workspace_dir": str(state), "task": task_text,
        "max_cycles": settings["max_cycles"],
        "llm": {"provider": "litellm", "model": settings["model"],
                "reasoning_effort": settings["reasoning_effort"]},
        # Arbor's shipped keyless backend, which needs no endpoint and no API key. Native
        # search.enabled is already true; with builtin_backend at its "none" default no
        # backend resolves, so the tools are simply never registered.
        "search": {"enabled": settings["retrieval"], "builtin_backend": "alphaxiv"},
        # A queued measurement prints nothing; native RunTraining must not treat
        # that silence as a hung job and kill the evaluation's process group.
        "timeout": {"run_training_stall": None},
        "protected_paths": list(benchmark.task["substrate"]["immutable"]),
    }
    metric = target(benchmark.task)
    reference = project(benchmark.reference_result, task_object(benchmark))
    metadata = {"baseline_score": reference["score"], "trunk_score": reference["score"],
                "test_baseline_score": reference["score"], "test_trunk_score": reference["score"],
                "metric_direction": metric["direction"], "eval_cmd": command,
                "dataset_info": task_text}
    return config, metadata


def prepare(benchmark, settings):
    config, metadata = configuration(benchmark, settings)
    code = Path(config["cwd"])
    if code.exists():
        raise RuntimeError("Native workspace already exists; use resume")
    shutil.copytree(benchmark.task_dir / "code", code)
    (code / ".gitignore").write_text(".arbor/\n.autoresearch/\n.coordinator/\nresults/\nresearch_log.json\n__pycache__/\n")
    for command in (["git", "init", "-b", "main"],
                    ["git", "config", "user.name", "AutoArena"],
                    ["git", "config", "user.email", "autoarena@example.com"],
                    ["git", "add", "."], ["git", "commit", "-m", "Frozen task reference"],
                    ["git", "branch", "research/arena/trunk"]):
        subprocess.run(command, cwd=code, check=True, stdout=subprocess.DEVNULL)
    cache = Path(config["workspace_dir"]) / ".coordinator"
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / "baseline_cache.json", metadata)
    store_reference(benchmark)
    import yaml
    path = benchmark.workspace / "arbor-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    benchmark = Benchmark.from_context()
    settings = json.loads((ROOT / "settings.json").read_text())
    if args.resume:
        path, code = benchmark.workspace / "arbor-config.yaml", benchmark.workspace / "arbor-code"
    else:
        path, code = prepare(benchmark, settings)
    command = [sys.executable, "-m", "arbor.coordinator.main", "--config", str(path),
               "--cwd", str(code), "--branch-prefix", "research/arena",
               "--trunk-branch", "research/arena/trunk"]
    if args.resume:
        command.extend(["--resume", "--allow-non-base-branch"])
    # Native scratch stays with this run.
    scratch = benchmark.workspace / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, TMPDIR=str(scratch))
    # Only the outside engine may execute task measurements on GPUs.
    command = ["bwrap", "--die-with-parent", "--bind", "/", "/",
               "--dev", "/dev", "--proc", "/proc", "--", *command]
    completed = subprocess.run(command, env=environment)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    benchmark.finish("Native Arbor coordinator completed; see native reports and tree")


if __name__ == "__main__":
    main()
