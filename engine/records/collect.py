"""Derive collected results, accounting and integrity findings from run evidence.

The summary follows `autoarena/collected/1`. Task rules determine admissibility
and ranking; the captured method interface supplies identity only.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from engine.evaluation.task import Task
from engine.evaluation.objective import admissibility, gate_margin, rank_key

from engine.records.ledger import LaunchView
from engine.records.runs import Run

SCHEMA = "autoarena/collected/1"


def collect(
    run: Run,
    *,
    task: Optional[Task] = None,
    views: Optional[List[LaunchView]] = None,
) -> Dict[str, Any]:
    """Fold one run into a summary dict. Never raises on a merely bad run.

    Malformed ledger rows raise while reading or folding. Crashes, overspending
    and missing measurements instead appear in the summary.
    `views` may supply an already validated canonical snapshot, including [].
    Only None reads and folds the run's ledger.
    """
    task = task or run.task()
    if views is None:
        views = run.views()

    launches = [_launch(task, view, index) for index, view in enumerate(views, start=1)]
    charged = [entry for entry in launches if entry["charged"]]
    admissible = [entry for entry in launches if entry["admissible"]]

    best_index, best = _best(task, launches)
    budget = _budget(task, launches)
    reference = _reference(task, launches)

    summary: Dict[str, Any] = {
        "schema": SCHEMA,
        "run": {
            "run_id": run.run_id,
            "path": str(run.path),
            "created_at": run.meta.get("created_at"),
            "created_by": run.meta.get("created_by"),
            "hostname": run.meta.get("hostname"),
            "lanes": run.meta.get("lanes"),
            "research_model": run.meta.get("research_model"),
            "compute": run.meta.get("compute") or {},
            "harness_revision": run.meta.get("harness_revision"),
            "notes": run.meta.get("notes", ""),
        },
        "task": {
            "task_id": task.task_id,
            "version": task.version,
            "title": task.title,
            "comparison_mode": task.comparison_mode,
            "headline": task.headline_metric,
            "headline_direction": task.headline_direction,
            "columns": list(task.report["columns"]),
            "ranking_metrics": [entry["metric"] for entry in task.ranking_metrics],
            "objective_metric": task.objective_spec["target"]["metric"],
            "gate": task.objective_spec.get("quality_gate"),
            "admissibility": task.objective_spec.get("constraints", []),
        },
        "method": _method_block(run),
        "budget": budget,
        "launches": launches,
        "best": best,
        "best_launch_seq": best_index,
        "reference": reference,
        "trajectory": _trajectory(task, launches),
        "status_counts": _counts(launches, "status"),
        "totals": {
            "launches_recorded": len(launches),
            "charged": len(charged),
            "admissible": len(admissible),
            "measured": sum(1 for entry in launches if entry["status"] == "ok"),
            "distinct_candidates": len({entry["candidate_id"] for entry in launches}),
            "gpu_seconds": round(
                sum(entry["wall_time_seconds"] or 0.0 for entry in launches), 1
            ),
        },
        "integrity": integrity(run, task, launches, budget=budget, reference=reference),
    }
    return summary


# ---------------------------------------------------------------------------
# per launch
# ---------------------------------------------------------------------------


def _launch(task: Task, view: Any, index: int) -> Dict[str, Any]:
    metrics = view.metrics
    charge = view.charge()
    verdict = admissibility(task, metrics) if view.resolved else None
    result = view.result or {}
    return {
        "index": index,
        "launch_seq": view.launch_seq,
        "launch_uuid": view.launch_uuid,
        "candidate_id": view.candidate_id,
        "parent_id": view.parent_id,
        "is_reference_launch": view.is_reference_launch,
        "lane": view.intent.get("lane"),
        "compute_run_id": view.intent.get("compute_run_id"),
        "started_at": view.intent.get("started_at"),
        "finished_at": result.get("finished_at"),
        "wall_time_seconds": result.get("wall_time_seconds"),
        "status": view.status,
        "resolved": view.resolved,
        "witness": bool(result.get("witness")),
        "exit_code": result.get("exit_code"),
        "charged": charge.charged,
        "charge_reason": charge.reason,
        "charge_classification": charge.classification,
        "adjudicated": bool(view.adjudications),
        # `admissible` is False for an unresolved launch, and `clause` says why. An
        # unresolved launch is not admissible-unknown-so-probably-fine.
        "admissible": bool(verdict is not None and verdict.admissible),
        "clause": verdict.clause if verdict is not None else "unresolved",
        "reason": verdict.reason if verdict is not None else "no result row yet",
        "gate_margin": gate_margin(task, metrics) if view.resolved else None,
        "rank_key": list(rank_key(task, metrics) or []) or None,
        "metrics": {name: metrics.get(name) for name in task.report["columns"]},
        "all_metrics": metrics,
        "extraction_errors": list(result.get("extraction_errors") or []),
        "idea": view.intent.get("idea"),
    }


def _best(task: Task, launches: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[Dict]]:
    """The best admissible launch by the task's own ranking order.

    Only entries with a rank key are considered. An inadmissible candidate is *absent*
    from this comparison rather than ranked last: on a minimized axis a ceiling breacher
    sits exactly where the most attractive values are, so leaving it in the pool wins.
    """
    ranked = [entry for entry in launches if entry["rank_key"] is not None]
    if not ranked:
        return None, None
    winner = min(ranked, key=lambda entry: tuple(entry["rank_key"]))
    return winner["launch_seq"], {
        "launch_seq": winner["launch_seq"],
        "candidate_id": winner["candidate_id"],
        "metrics": winner["metrics"],
        "headline": winner["metrics"].get(task.headline_metric),
        "gate_margin": winner["gate_margin"],
        "rank_key": winner["rank_key"],
    }


def _trajectory(task: Task, launches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-so-far after each charged launch.

    Indexed by *charged* launch, not by wall-clock or by row: charged launches are the
    budget, and a chart plotted against row index rewards a method whose refunded
    substrate failures pad the x axis.
    """
    points: List[Dict[str, Any]] = []
    spent = 0
    best: Optional[Tuple] = None
    best_value: Optional[float] = None
    for entry in launches:
        if not entry["charged"]:
            continue
        spent += 1
        key = tuple(entry["rank_key"]) if entry["rank_key"] is not None else None
        if key is not None and (best is None or key < best):
            best = key
            best_value = entry["metrics"].get(task.headline_metric)
        points.append(
            {
                "spent": spent,
                "launch_seq": entry["launch_seq"],
                "status": entry["status"],
                "admissible": entry["admissible"],
                "value": entry["metrics"].get(task.headline_metric),
                "best_so_far": best_value,
            }
        )
    return points


def _counts(launches: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in launches:
        counts[str(entry[field])] = counts.get(str(entry[field]), 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def _budget(task: Task, launches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The budget, derived. `distinct launch_uuid` is the unit, and always was."""
    charged = {entry["launch_uuid"] for entry in launches if entry["charged"]}
    refunded = {entry["launch_uuid"] for entry in launches if not entry["charged"]}
    unresolved = {entry["launch_uuid"] for entry in launches if not entry["resolved"]}
    cap = task.max_launches
    return {
        "max_launches": cap,
        "includes_failures": bool(task.budget.get("includes_failures")),
        "includes_reference_launch": bool(task.budget.get("includes_reference_launch")),
        "charging_rule": task.budget.get("charging_rule"),
        "spent": len(charged),
        "refunded": len(refunded),
        "unresolved": len(unresolved),
        "remaining": cap - len(charged),
        "overspent": len(charged) > cap,
        "refund_reasons": _counts(
            [entry for entry in launches if not entry["charged"]], "charge_classification"
        ),
    }


# ---------------------------------------------------------------------------
# reference launch
# ---------------------------------------------------------------------------


def _reference(task: Task, launches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare the first reference launch with the task's exact assertions."""
    found = [entry for entry in launches if entry["is_reference_launch"]]
    expected = task.reference_launch.get("assert_exact") or {}
    block: Dict[str, Any] = {
        "expected_position": task.reference_launch.get("position"),
        "assert_exact": expected,
        "present": bool(found),
        "launch_seq": found[0]["launch_seq"] if found else None,
        "count": len(found),
        "mismatches": [],
        "recorded": {},
        "reproduced": False,
    }
    if not found:
        return block
    entry = found[0]
    block["recorded"] = entry["all_metrics"]
    for metric, want in expected.items():
        got = entry["all_metrics"].get(metric)
        if got is None or float(got) != float(want):
            block["mismatches"].append({"metric": metric, "expected": want, "observed": got})
    block["reproduced"] = entry["status"] == "ok" and not block["mismatches"]
    return block


# ---------------------------------------------------------------------------
# method block
# ---------------------------------------------------------------------------


def _method_block(run: Run) -> Dict[str, Any]:
    """Identify the measured method using its captured interface."""
    interface = (run.meta.get("method_snapshot") or {}).get("engine_interface")
    return {
        "method_id": run.method_id,
        "title": interface.get("id") if isinstance(interface, dict) else None,
    }


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------


def integrity(
    run: Run,
    task: Task,
    launches: List[Dict[str, Any]],
    *,
    budget: Dict[str, Any],
    reference: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Findings that change how a row should be read. Each one has a level.

    `blocking` means the number is not comparable and the report says so next to it.
    `warn` means read it with the finding in view. Nothing here is cosmetic-only, because
    a finding nobody acts on is noise that trains readers to skip the section.
    """
    findings: List[Dict[str, Any]] = []

    def add(level: str, code: str, detail: str) -> None:
        findings.append({"level": level, "code": code, "detail": detail})

    compute = run.meta.get("compute") or {}
    description = compute.get("describe") or {}
    if isinstance(description, dict) and description.get("test_only") is True:
        add(
            "blocking",
            "test_only_compute",
            "the compute backend declares test_only=true. Its measurements are fixtures, "
            "so this run exercises the pipeline but cannot enter a benchmark ranking",
        )

    # -- the record itself -------------------------------------------------
    unresolved = [entry for entry in launches if not entry["resolved"]]
    if unresolved:
        add(
            "blocking",
            "unresolved_launches",
            f"{len(unresolved)} launch(es) have an intent row and no result: seq "
            + ", ".join(str(entry["launch_seq"]) for entry in unresolved[:10])
            + ". Each is charged conservatively, so the spend is right and the outcome is "
            "unknown. Adjudicate them or rerun the tail",
        )

    seqs = [entry["launch_seq"] for entry in launches]
    if len(set(seqs)) != len(seqs):
        add(
            "blocking",
            "duplicate_launch_seq",
            "two intent rows share a launch_seq, which means two writers shared the "
            "ledger. Under concurrency that also interleaves partial lines",
        )

    # -- one measurement per candidate ------------------------------------
    measured: Dict[str, List[int]] = {}
    for entry in launches:
        if entry["status"] == "ok":
            measured.setdefault(entry["candidate_id"], []).append(entry["launch_seq"])
    repeats = {name: seq for name, seq in measured.items() if len(seq) > 1}
    if repeats:
        add(
            "warn",
            "candidate_measured_twice",
            "candidate(s) measured more than once: "
            + "; ".join(f"{name} at {seq}" for name, seq in sorted(repeats.items())[:5])
            + ". State deliberate replicates in the run notes",
        )

    # -- the reference -----------------------------------------------------
    if not reference["present"]:
        add(
            "blocking",
            "no_reference_launch",
            "no launch is marked is_reference_launch. Without it nothing establishes that "
            "this machine reproduces the substrate, and every candidate number inherits "
            "the unmeasured discrepancy",
        )
    else:
        if reference["mismatches"]:
            add(
                "blocking",
                "reference_mismatch",
                "the reference launch did not reproduce the task's asserted values: "
                + "; ".join(
                    f"{item['metric']} expected {item['expected']}, observed "
                    f"{item['observed']}"
                    for item in reference["mismatches"]
                )
                + ". These are deterministic given the source, so this is a different "
                "substrate, not run-to-run variation",
            )
        position = reference["expected_position"]
        if position is not None and reference["launch_seq"] != position:
            add(
                "warn",
                "reference_out_of_position",
                f"the reference launch is at seq {reference['launch_seq']}, and the task "
                f"expects position {position}. Later is worse than never only in that a "
                "mismatch is discovered after the budget is spent",
            )

    # -- the budget --------------------------------------------------------
    if budget["overspent"]:
        add(
            "blocking",
            "budget_overspent",
            f"{budget['spent']} charged launches against a cap of {budget['max_launches']}. "
            "The extra launches are not comparable to an arm that stopped at the cap, and "
            "the best-so-far is the best over a larger search",
        )

    # -- measurement quality ----------------------------------------------
    ranking = set(summary_metrics(task))
    broken = [
        entry
        for entry in launches
        if entry["status"] == "ok"
        and any(any(name in error for name in ranking) for error in entry["extraction_errors"])
    ]
    if broken:
        add(
            "warn",
            "extraction_errors_on_ranking_metrics",
            f"{len(broken)} launch(es) exited cleanly but a metric that ranks them could "
            "not be read. They are charged and cannot be compared",
        )

    # -- the task itself ---------------------------------------------------
    snapshot = run.meta.get("task_snapshot")
    if snapshot is None:
        add(
            "warn",
            "no_task_snapshot",
            "run.json records no task_snapshot, so a change to the task since the run "
            "cannot be detected and this row may have been scored under other rules",
        )
    elif json.dumps(snapshot, sort_keys=True) != json.dumps(task.raw, sort_keys=True):
        add(
            "blocking",
            "task_changed_since_run",
            f"the definition of {task.task_id} has changed since this run started. The row "
            "was scored under the recorded snapshot and the table is rendered under the "
            "current task; rerun, or report it beside runs of the same snapshot only",
        )

    return findings


def summary_metrics(task: Task) -> List[str]:
    """The metrics that decide something: the ranking order plus every ceiling."""
    names = [entry["metric"] for entry in task.ranking_metrics]
    names.append(task.objective_spec.get("quality_gate", task.objective_spec["target"])["metric"])
    names.extend(entry["metric"] for entry in task.objective_spec.get("constraints", []))
    seen, ordered = set(), []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered
