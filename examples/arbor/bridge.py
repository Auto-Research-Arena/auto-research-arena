"""Canonical evaluation and source-bound feedback for native Arbor."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from autoarena import Benchmark, objective_spec


def task_object(benchmark):
    sys.path.insert(0, benchmark.context["runner_root"])
    from engine.evaluation.task import Task, validate
    raw = benchmark.task
    path = benchmark.task_dir / "task.json"
    validate(raw, path)
    return Task(task_id=raw["task_id"], version=raw["version"],
                title=raw["title"], path=path, raw=raw)


def target(task):
    return objective_spec(task["objective"])["target"]


def project(result, task, reference=None):
    """Keep the raw target; eligibility does not invent a penalty score."""
    from engine.evaluation.objective import admissibility
    metrics = result.get("metrics", {})
    verdict = admissibility(task, metrics)
    eligible = result.get("status") == "ok" and verdict.admissible
    return {"score": metrics.get(task.objective_spec["target"]["metric"]),
            "eligible": eligible, "reason": verdict.reason if result.get("status") == "ok"
            else "Canonical evaluation did not succeed", "metrics": metrics}


def identity(benchmark, source, branch=None):
    """Match the benchmark's exact declared-file identity, including deletions."""
    task = task_object(benchmark)
    from engine.runtime.lifecycle import encoded, program_identity
    if branch is None:
        return program_identity(task, Path(source))
    files = {}
    for name in sorted(set(task.substrate["mutable"] + task.substrate["immutable"])):
        try:
            payload = subprocess.check_output(["git", "show", f"{branch}:{name}"], cwd=source,
                                              stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise ValueError(f"Branch {branch} is missing declared task file {name}") from error
        files[name] = hashlib.sha256(payload).hexdigest()
    return {"files": files, "sha256": hashlib.sha256(encoded({"task": task.raw, "files": files})).hexdigest()}


def records(benchmark):
    directory = benchmark.workspace / "arbor-results"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def store_reference(benchmark):
    key = identity(benchmark, benchmark.task_dir / "code")
    row = {"identity": key, "result": benchmark.reference_result,
           "request_id": "reference"}
    write_json(records(benchmark) / (key["sha256"] + ".json"), row)


def evaluate(source, research_log, benchmark=None):
    benchmark = benchmark or Benchmark.from_context()
    source = Path(source).resolve()
    key = identity(benchmark, source)
    directory = records(benchmark)
    # One measurement worker; native executors can still prepare in parallel.
    with (directory / "measurement.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / (key["sha256"] + ".json")
        row = json.loads(path.read_text()) if path.exists() else None
        if row is None:
            if not research_log.get("ideas") or not research_log.get("status"):
                raise ValueError("research_log requires nonempty ideas and status")
            snapshot = directory / key["sha256"] / "code"
            snapshot.mkdir(parents=True, exist_ok=True)
            for name in key["files"]:
                destination = snapshot / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, destination)
            if identity(benchmark, snapshot) != key:
                raise RuntimeError("Source changed while preparing evaluation")
            row = {"identity": key, "request_id": "arbor-" + key["sha256"],
                   "source": str(snapshot), "research_log": research_log}
            # Persist exact submission inputs before making an API request.
            write_json(path, row)
        if "result" not in row:
            row["result"] = benchmark.evaluate(row["source"], row["request_id"],
                                               research_log=row["research_log"])
            write_json(path, row)
    result = project(row["result"], task_object(benchmark))
    return {**result, "request_id": row["request_id"], "identity": key,
            "canonical_result": row["result"]}


def recorded_branch(cwd, branch, benchmark=None):
    benchmark = benchmark or Benchmark.from_context()
    key = identity(benchmark, cwd, branch)
    path = records(benchmark) / (key["sha256"] + ".json")
    if not path.exists():
        raise ValueError(f"Branch {branch} has no canonical measurement for its current task code")
    row = json.loads(path.read_text())
    if row.get("identity") != key or "result" not in row:
        raise ValueError(f"Branch {branch} has no completed canonical result")
    return {**project(row["result"], task_object(benchmark)),
            "identity": key, "request_id": row["request_id"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--research-log", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.source, json.loads(args.research_log.read_text()))
    print(json.dumps(result, indent=2))
    # Native score parsers can consume the target without rounded prose.
    print(json.dumps({"score": result["score"], "eligible": result["eligible"]}))


if __name__ == "__main__":
    main()
