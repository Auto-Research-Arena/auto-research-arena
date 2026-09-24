"""Validate task packages and method interfaces."""

import json
from pathlib import Path

from engine import ROOT
from engine.evaluation.task import TaskError, available_tasks, load_task


def check(root=ROOT):
    from engine.runtime.lifecycle import EngineError, interface, _launch_prompt
    failures, task_ids, method_ids = [], available_tasks(root / "tasks"), []
    if not task_ids:
        failures.append("no task definitions found")
    for task_id in task_ids:
        try:
            task = load_task(task_id, root / "tasks")
            code = task.source_root
            for name in task.substrate["mutable"] + task.substrate["immutable"]:
                target = code / name
                if Path(name).is_absolute() or ".." in Path(name).parts or target.is_symlink():
                    raise ValueError(f"task file is not a regular package path: {name}")
                if not target.is_file():
                    raise ValueError(f"missing task file: {name}")
        except (OSError, ValueError, TaskError) as error:
            failures.append(f"{task_id}: {error}")
    for location in sorted((root / "examples").glob("*/interface.json")):
        try:
            method = interface(json.loads(location.read_text()), template=True)
            if method["id"] != location.parent.name:
                raise ValueError("interface id differs from its folder")
            _launch_prompt(method, location.parent)
            method_ids.append(method["id"])
        except (OSError, ValueError, EngineError) as error:
            failures.append(f"{location}: {error}")
    if not method_ids:
        failures.append("no method interfaces found")
    rows = [{"name": "Task packages", "status": "ready" if task_ids else "failed",
             "detail": f"{len(task_ids)} task definitions and their declared source files"},
            {"name": "Method interfaces", "status": "ready" if method_ids else "failed",
             "detail": f"{len(method_ids)} launch interfaces and prompt assets"}]
    rows += [{"name": "Package check", "status": "failed", "detail": error} for error in failures]
    rows.append({"name": "GPU, environment and LLM probes", "status": "skipped",
                 "detail": "select a run with --config to check its local setup"})
    return {"tasks": task_ids, "methods": method_ids, "failures": failures,
            "checks": rows, "passed": not failures}
