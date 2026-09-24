"""Read and audit one run for submission assembly."""

import json
import math
from pathlib import Path

from engine.records import collect, runs, verify
from engine.runtime import inputs
from engine.evaluation.task import Task, validate


def _stamp(path):
    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except FileNotFoundError:
        return None


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _comparison(task, collection):
    references = [row for row in collection["launches"] if row["is_reference_launch"]]
    reference = references[0] if len(references) == 1 else None
    best = collection["best"]
    metric = task.headline_metric
    ref_value = _number(reference["all_metrics"].get(metric)) if reference else None
    best_value = _number(best["headline"]) if best else None
    improvement = None
    if ref_value is not None and best_value is not None:
        improvement = (ref_value - best_value if task.headline_direction == "minimize"
                       else best_value - ref_value)
    return {
        "metric": metric,
        "unit": task.metrics[metric].get("unit"),
        "direction": task.headline_direction,
        "reference_value": ref_value,
        "reference_admissible": reference["admissible"] if reference else None,
        "best_value": best_value,
        "best_is_reference": (best["launch_seq"] == reference["launch_seq"]
                              if best and reference else None),
        "absolute_improvement": improvement,
        "relative_improvement": (improvement / abs(ref_value)
                                 if improvement is not None and ref_value else None),
        "strictly_better_than_reference": (
            tuple(best["rank_key"]) < tuple(reference["rank_key"])
            if best and reference and reference["rank_key"] is not None else None),
    }


def _run_report(directory):
    controls = [directory / name for name in (
        "run.json", "launches.jsonl", "engine/definition.json", "engine/state.json",
        "engine/finished.json", "engine/stop.json")]
    before = {path: _stamp(path) for path in controls}
    run = runs.open_run(directory)
    raw = run.meta.get("task_snapshot")
    if not isinstance(raw, dict):
        raise ValueError("a final report requires the run's recorded task_snapshot")
    validate(raw)
    if run.meta.get("task_snapshot_sha256") != runs._json_sha256(raw):
        raise ValueError("recorded task snapshot does not match its stored identity")
    method = run.meta.get("method_snapshot")
    if method is not None and run.meta.get("method_snapshot_sha256") != runs._json_sha256(method):
        raise ValueError("recorded method snapshot does not match its stored identity")
    if raw["task_id"] != run.task_id or raw["version"] != run.meta.get("task_version"):
        raise ValueError("recorded task identity differs from the run")
    # Report historical runs under their recorded rules, never today's renamed task.
    task = Task(raw["task_id"], raw["version"], raw["title"], directory / "task.json", raw)
    views = run.views()
    for view in views:
        for key in ("stdout_path", "stderr_path"):
            value = (view.result or {}).get(key)
            if value:
                path = runs.log_path(run, value)
                before[path] = _stamp(path)
        for name in task.substrate["mutable"] + task.substrate["immutable"]:
            path = runs.candidate_source(run, view.candidate_id) / name
            before[path] = _stamp(path)
    collection = collect.collect(run, task=task, views=views)
    audit = verify.verify(run, task=task, views=views).to_dict()
    recorded = {}
    for name in ("definition", "state", "finished"):
        path = directory / f"engine/{name}.json"
        value = json.loads(inputs.read_captured_file(path)) if path.exists() else {}
        if not isinstance(value, dict):
            raise ValueError(f"recorded {name} is malformed")
        recorded[name] = value
    state = recorded["state"]
    if not isinstance(state.get("status", "unknown"), str):
        raise ValueError("recorded lifecycle state is malformed")
    lifecycle = {
        "recorded_status": state.get("status", "unknown"),
        "method_declared_complete": (directory / "engine/finished.json").is_file(),
        "stop_requested": (directory / "engine/stop.json").is_file(),
    }
    if any(_stamp(path) != stamp for path, stamp in before.items()):
        raise ValueError("run evidence changed while reporting; stop writers and retry")
    unresolved = collection["budget"]["unresolved"]
    incomplete = (not views or unresolved or
                  audit["checked"].get("stdout_reread", 0) != len(views) or
                  any(row["code"].startswith("unverifiable_") for row in audit["findings"]))
    evidence_status = "failed" if not audit["passed"] else "incomplete" if incomplete else "verified"
    report = {
        "collection": collection,
        "verification": audit,
        "task_definition": raw,
        "lifecycle": lifecycle,
        "evidence_status": evidence_status,
        "comparison": _comparison(task, collection),
    }
    return report, run, views, recorded


def _strict_json(value, changes, path=""):
    """JSON has no NaN/Infinity; retain their locations rather than invent scores."""
    if isinstance(value, float) and not math.isfinite(value):
        changes.append({"path": path, "value": str(value)})
        return None
    if isinstance(value, dict):
        return {key: _strict_json(item, changes, path + "/" + key.replace("~", "~0").replace("/", "~1"))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_strict_json(item, changes, path + "/" + str(index))
                for index, item in enumerate(value)]
    return value


def build(run_directory):
    """Return audited measurements for the submission builder."""
    data, _, _, _ = _run_report(Path(run_directory).resolve())
    changes = []
    data = _strict_json(data, changes)
    data["nonfinite_values"] = changes
    return data
