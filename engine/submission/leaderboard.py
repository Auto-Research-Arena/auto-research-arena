"""Read website result records and build the public leaderboard."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from engine import ROOT
from engine.evaluation.task import available_tasks, load_task

MAX_JSON_BYTES = 20 * 1024 * 1024
ELO_SHUFFLES = 4000
ELO_SEED = 20260902
ELO_K = 32.0


class LeaderboardError(ValueError):
    """A public submission cannot be included in the leaderboard."""


def require(condition, message):
    if not condition:
        raise LeaderboardError(message)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_json(raw: bytes):
    def invalid(value):
        raise LeaderboardError(f"non-standard JSON number: {value}")

    try:
        return json.loads(raw, parse_constant=invalid)
    except (ValueError, UnicodeError) as error:
        raise LeaderboardError(f"invalid JSON: {error}") from error


def public_url(url: str) -> str:
    """Public artifacts use HTTPS URLs without embedded credentials."""
    require(isinstance(url, str), "artifact URL must be a string")
    parts = urlsplit(url)
    require(parts.scheme == "https" and parts.hostname, "artifact URL must use HTTPS")
    require(not parts.username and not parts.password and not parts.query and not parts.fragment,
             "use an artifact URL without credentials, query parameters or fragments")
    require(not any(c.isspace() or ord(c) < 32 for c in url), "invalid artifact URL")
    return url


def read_submission(path: Path) -> bytes:
    raw = path.read_bytes()
    require(len(raw) <= MAX_JSON_BYTES, "submission.json exceeds 20 MiB")
    return raw


def validate_artifacts(links: dict) -> dict:
    require(isinstance(links, dict), "artifacts.json must contain an object")
    require(links.keys() <= {"code_url", "logs_url", "report_url"},
            "artifacts.json supports code_url, logs_url and optional report_url")
    return {name: public_url(url) for name, url in links.items()}


def site_record(record: dict, *, method_id=None, task_id=None, run_id=None) -> dict:
    """Keep recorded results in the website's common JSON format."""
    headline = record["headline"]
    task = load_task(task_id or headline["task_id"])
    setup = record.get("setup", {})
    fields = ("metric", "unit", "direction", "best_value", "reference_value",
              "relative_improvement", "rank_key", "best_launch_seq",
              "best_candidate_id", "best_is_reference")
    return {
        "schema": "autoarena/site-submission/1",
        "run_id": run_id or record["identity"]["run_id"],
        "method_id": method_id or headline["method_id"],
        "task_id": task.task_id,
        **{key: headline[key] for key in fields},
        "budget": {key: record["cost"]["budget"][key] for key in ("spent", "max_launches")},
        "compute": setup.get("compute", {}),
        "research_llm": {key: value for key, value in setup.get("research_llm", {}).items()
                         if key in ("model", "provider", "used")},
        "evaluation_gpu_hours": record["cost"].get("evaluation_gpu_hours"),
        "measurements": [{
            "launch_seq": row["launch_seq"], "candidate_id": row["candidate_id"],
            "is_reference": row["is_reference"], "charged": row["charged"],
            "qualifying": row["rank_key"] is not None,
            "metrics": {name: row["metrics"].get(name) for name in task.metrics},
        } for row in record["history"]],
    }


def project_site_record(record: dict) -> dict:
    """Check result consistency and derive chart points from recorded measurements."""
    require(record.get("schema") == "autoarena/site-submission/1",
            "expected autoarena/site-submission/1")
    for key in ("run_id", "method_id", "task_id"):
        require(isinstance(record.get(key), str) and record[key].strip(), f"missing {key}")
    task = load_task(record["task_id"])
    require(record["metric"] == task.headline_metric
            and record["direction"] == task.headline_direction,
            "result metric or direction differs from the target")
    for key in ("best_value", "reference_value"):
        require(isinstance(record[key], (int, float)) and not isinstance(record[key], bool)
                and math.isfinite(record[key]), f"{key} must be finite")
    measurements = record["measurements"]
    sequences = [row["launch_seq"] for row in measurements]
    require(sequences and sequences == sorted(set(sequences)),
            "measurements must have unique, increasing launch sequences")
    points, curve, step, best = [], [], 0, None
    metric = record["metric"]
    for row in measurements:
        require(isinstance(row["metrics"], dict) and row["metrics"].keys() == task.metrics.keys(),
                "measurement metric fields must match the target definition")
        require(all(type(row[key]) is bool for key in ("charged", "is_reference", "qualifying")),
                "measurement flags must be booleans")
        step += int(row["charged"])
        if row["qualifying"]:
            value = row["metrics"].get(metric)
            require(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value), "qualifying measurements need a finite target value")
            points.append({"step": step, "launch_seq": row["launch_seq"],
                           "candidate_id": row["candidate_id"], "value": value,
                           "is_reference": row["is_reference"]})
            better = best is None or (value < best if record["direction"] == "minimize" else value > best)
            if better:
                best = value
        if row["charged"]:
            curve.append({"step": step, "value": best})
    budget = record["budget"]
    require(step == budget["spent"] and 0 <= step <= budget["max_launches"],
            "budget differs from the measurement count")
    require(best == record["best_value"], "best value differs from the recorded measurements")
    winner = next((row for row in measurements
                   if row["launch_seq"] == record["best_launch_seq"]), None)
    require(winner is not None and winner["candidate_id"] == record["best_candidate_id"]
            and winner["metrics"].get(metric) == best,
            "winning measurement differs from the result")
    reference = next((row for row in measurements if row["is_reference"]), None)
    require(reference is not None and reference["metrics"].get(metric) == record["reference_value"],
            "reference value differs from the measurement")
    require(isinstance(record["rank_key"], list) and record["rank_key"]
            and all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in record["rank_key"]), "ranking key must contain finite numbers")
    return {**{key: value for key, value in record.items() if key not in ("schema", "measurements")},
            "points": points, "curve": curve}


def check_submission(path: Path) -> dict:
    """Read a website result and its optional artifact links."""
    raw = read_submission(path)
    result = project_site_record(read_json(raw))
    links = validate_artifacts(read_json(path.with_name("artifacts.json").read_bytes()))
    return {**result, "submission_sha256": digest(raw),
            "code_url": None, "logs_url": None, "report_url": None, **links}


def _margin_elo(methods: list[str], comparisons: list, weights: dict) -> tuple[dict, str]:
    """Equal-target Elo with normalized log margins and averaged match orders."""
    import numpy as np

    ratings = {method: {"elo": None, "match_order_sd": None} for method in methods}
    neighbors = {method: set() for method in methods}
    for _, left, right, _ in comparisons:
        neighbors[left].add(right)
        neighbors[right].add(left)
    active = [method for method in methods if neighbors[method]]
    if not active:
        return ratings, "no_comparisons"
    reached, pending = set(), [active[0]]
    while pending:
        method = pending.pop()
        if method not in reached:
            reached.add(method)
            pending.extend(neighbors[method] - reached)
    if len(reached) != len(active):
        # Independent sets of comparisons do not establish a global ordering.
        return ratings, "disconnected"
    if any(margin is None for _, _, _, margin in comparisons):
        # A log ratio cannot rate zero/negative endpoints; retain their task results.
        return ratings, "nonpositive_values"
    scales = {}
    for target in weights:
        margins = [abs(margin) for task, _, _, margin in comparisons
                   if task == target and margin != 0]
        scales[target] = max(float(np.median(margins)), 1e-9) if margins else 1.0
    indices = {method: index for index, method in enumerate(active)}
    matches = []
    for target, left, right, margin in comparisons:
        outcome = 0.5 if margin == 0 else float(margin > 0)
        multiplier = min(3.0, math.log1p(abs(margin) / scales[target]) / math.log(2.0))
        matches.append((indices[left], indices[right], outcome,
                        ELO_K * (weights[target] / (1.0 / len(weights))) * multiplier))
    rng = np.random.default_rng(ELO_SEED)
    base_order = np.arange(len(matches), dtype=np.int32)
    by_order = np.empty((ELO_SHUFFLES, len(active)), dtype=float)
    for shuffle in range(ELO_SHUFFLES):
        current = np.full(len(active), 1500.0, dtype=float)
        for index in rng.permutation(base_order):
            left, right, outcome, factor = matches[int(index)]
            expected = 1.0 / (1.0 + 10.0 ** ((current[right] - current[left]) / 400.0))
            update = factor * (outcome - expected)
            current[left] += update
            current[right] -= update
        by_order[shuffle] = current
    mean = by_order.mean(axis=0)
    deviation = by_order.std(axis=0)
    for method, value, sd in zip(active, mean, deviation):
        ratings[method] = {"elo": float(value), "match_order_sd": float(sd)}
    return ratings, "ready"


def summarize_methods(tasks: list[dict], policy: dict | None = None) -> dict:
    """One matrix row per method, using its best accepted run on each task."""
    if policy is None:
        policy = read_json((ROOT / "site/elo.json").read_bytes())
    by_id = {task["id"]: task for task in tasks}
    require(by_id.keys() <= set(policy["target_order"]),
            "register each target in site/elo.json target_order")
    tasks = [by_id[key] for key in policy["target_order"]
             if key in by_id and by_id[key]["runs"]]
    weights = {task["id"]: 1.0 / len(tasks) for task in tasks}
    present_methods = {run["method_id"] for task in tasks for run in task["runs"]}
    methods = [method for method in policy["method_order"] if method in present_methods]
    methods += sorted(present_methods - set(methods))
    rows = {method: {"method_id": method, "cells": {}, "coverage": 0, "matches": 0}
            for method in methods}
    comparisons = []
    for task in tasks:
        best = {}
        for run in sorted(task["runs"], key=lambda run: (run["rank_key"], run["run_id"])):
            best.setdefault(run["method_id"], run)
        previous, rank = None, 0
        ranked = sorted(best.items(), key=lambda item:
                        (item[1]["comparison_rank_key"], item[1]["run_id"]))
        for index, (method, run) in enumerate(ranked, 1):
            if run["comparison_rank_key"] != previous:
                rank = index
            previous = run["comparison_rank_key"]
            rows[method]["cells"][task["id"]] = {
                "rank": rank, "value": run["comparison_value"], "run_id": run["run_id"],
                "report_url": run.get("report_url"),
            }
            rows[method]["coverage"] += 1
        present = [method for method in methods if method in best]
        for i, left in enumerate(present):
            for right in present[i + 1:]:
                left_value, right_value = best[left]["comparison_value"], best[right]["comparison_value"]
                margin = None
                if left_value > 0 and right_value > 0:
                    margin = math.log(right_value / left_value)
                    if task["direction"] == "maximize":
                        margin = -margin
                comparisons.append((task["id"], left, right, margin))
                rows[left]["matches"] += 1
                rows[right]["matches"] += 1
    ratings, status = _margin_elo(methods, comparisons, weights)
    for method in methods:
        rows[method].update(ratings[method])
    ordered = sorted(rows.values(), key=lambda row:
                     (row["elo"] is None, -(row["elo"] or 0), row["method_id"]))
    previous, rank = None, None
    for index, row in enumerate(ordered, 1):
        if row["elo"] is not None and row["elo"] != previous:
            rank = index
        row["rank"] = rank if row["elo"] is not None else None
        previous = row["elo"]
    return {
        "methods": ordered, "status": status, "task_count": len(tasks),
        "comparison_count": len(comparisons),
        "rating": {
            "method": "Order-averaged margin-aware Elo",
            "base": 1500, "scale": 400, "cohort_mean": 1500, "leader_anchor": None,
            "shuffles": ELO_SHUFFLES, "seed": ELO_SEED, "k_factor": ELO_K,
            "method_order": methods, "target_order": [task["id"] for task in tasks],
            "weighting": "equal_per_included_target",
            "target_weights": weights,
            "rounding_steps": policy["rounding_steps"],
            "rounding_formula": "step * floor(raw_value / step + 0.5)",
            "margin": "min(3, log1p(abs(log(right / left)) / target_scale) / log(2))",
            "target_scale": "median nonzero absolute pairwise log gap, floored at 1e-9; 1 if all tied",
            "comparison_order": "configured target and method order; new methods appended by ID",
            "run_selection": "best accepted run per method and task",
            "missing_targets": "excluded from comparisons",
            "ties": "equal target values: outcome 0.5, zero margin and zero update",
            "match_order_sd": "population standard deviation across shuffled passes; order sensitivity",
        },
    }


def comparison_value(value, step):
    """Round display/comparison values while retaining raw submission measurements."""
    return step * math.floor(value / step + 0.5) if step and value is not None else value


def build_data(results: list[dict]) -> dict:
    """Aggregate checked local submissions, with no network access."""
    policy = read_json((ROOT / "site/elo.json").read_bytes())
    require(all(isinstance(step, (int, float)) and not isinstance(step, bool)
                and math.isfinite(step) and step > 0
                for step in policy["rounding_steps"].values()),
            "comparison rounding steps must be finite and positive")
    groups = {}
    for name in available_tasks():
        task = load_task(name)
        groups[name] = {
            "id": name, "title": task.title, "metric": task.headline_metric,
            "unit": task.metrics[task.headline_metric].get("unit"),
            "direction": task.headline_direction, "budget": task.max_launches,
            "comparison_step": policy["rounding_steps"].get(name),
            "quality_gate": task.objective_spec["quality_gate"],
            "tiebreaks": task.objective_spec["tiebreaks"], "runs": [],
        }
    identities = set()
    for checked in results:
        result = deepcopy(checked)
        name = result["task_id"]
        require(name in groups, f"unknown task: {name}")
        key = (name, result["run_id"])
        require(key not in identities, f"duplicate run: {name}/{result['run_id']}")
        identities.add(key)
        step = groups[name]["comparison_step"]
        value = comparison_value(result["best_value"], step)
        result["comparison_value"] = value
        result["comparison_rank_key"] = (
            [value if result["direction"] == "minimize" else -value]
            if step else result["rank_key"])
        for point in result["points"] + result["curve"]:
            point["value"] = comparison_value(point["value"], step)
        groups[name]["runs"].append(result)
    for group in groups.values():
        group["runs"].sort(key=lambda row: (row["comparison_rank_key"], row["run_id"]))
        previous, rank = None, 0
        for index, row in enumerate(group["runs"], 1):
            if row["comparison_rank_key"] != previous:
                rank = index
            row["rank"] = rank
            previous = row["comparison_rank_key"]
    tasks = list(groups.values())
    return {"schema": "autoarena/leaderboard/1", "tasks": tasks,
            "overall": summarize_methods(tasks, policy)}
