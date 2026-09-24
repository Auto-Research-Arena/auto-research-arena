"""Turn one run into a submission: the headline result and everything behind it."""

from __future__ import annotations

import json
from pathlib import Path

from engine.submission import data as submission_data
from engine.records import runs
from engine.evaluation import objective as objective_module
from engine.autoarena import objective_spec

SCHEMA = "autoarena/submission"

STATUS_MEANING = {
    "ok": "The instrument returned a finite headline measurement.",
    "oom": "The candidate ran out of memory.",
    "crash": "The candidate exited with an error.",
    "fail_loss": "The headline measurement was missing or non-finite.",
    "substrate_failure": "The command exited without the GPU-work witness.",
    "never_executed": "The launch never started.",
    "preempted": "The compute allocation was reclaimed.",
    "timeout": "The launch exceeded the task's wall-clock limit.",
    "unresolved": "An intent has no recorded outcome; charged conservatively.",
}


class SubmissionError(ValueError):
    """The run cannot be turned into a submission record."""


class _TaskView:
    """Objective attributes over a frozen task definition."""

    def __init__(self, definition: dict):
        raw = (definition or {}).get("objective") or {}
        try:
            self.objective_spec = objective_spec(raw)
        except (ValueError, KeyError, TypeError) as error:
            raise SubmissionError(f"cannot read this run's frozen objective: {error}") from error

    @property
    def ranking_metrics(self) -> list:
        spec = self.objective_spec
        return [spec["target"], *spec.get("tiebreaks", [])]


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def eligibility(definition: dict, metrics: dict | None) -> list:
    """Task-defined gates and constraints, with measured values beside their bounds."""
    view = _TaskView(definition)
    spec = view.objective_spec
    metrics = metrics or {}
    rows = []

    def satisfied(value, operator, bound):
        if value is None or bound is None:
            return None
        try:
            return objective_module.satisfies(float(value), operator, float(bound))
        except (TypeError, ValueError):
            return None

    gate = spec.get("quality_gate")
    if gate:
        rows.append({"kind": "quality_gate", "metric": gate.get("metric"),
                     "operator": gate.get("operator"), "bound": gate.get("value"),
                     "measured": metrics.get(gate.get("metric")),
                     "satisfied": satisfied(metrics.get(gate.get("metric")),
                                            gate.get("operator"), gate.get("value")),
                     "basis": gate.get("basis")})
    for clause in spec.get("constraints") or []:
        rows.append({"kind": "constraint", "metric": clause.get("metric"),
                     "operator": clause.get("operator"), "bound": clause.get("value"),
                     "measured": metrics.get(clause.get("metric")),
                     "satisfied": satisfied(metrics.get(clause.get("metric")),
                                            clause.get("operator"), clause.get("value")),
                     "basis": clause.get("basis")})
    return rows


def obligations(definition: dict) -> dict:
    """The rules that are prose rather than arithmetic, and are still binding."""
    substrate = (definition or {}).get("substrate") or {}
    return {
        "known_constraints": list(substrate.get("known_constraints") or []),
        "frozen_regions": [
            {"file": region.get("file"), "region": region.get("region"),
             "anchor": region.get("anchor"), "reason": region.get("reason")}
            for region in substrate.get("frozen_regions") or []
        ],
        "mutable_files": list(substrate.get("mutable") or []),
        "immutable_files": list(substrate.get("immutable") or []),
    }


def verdict(run_report: dict, agrees: bool) -> dict:
    """Whether this run may be published, with every reason, and caveats that did not decide it."""
    collection = run_report.get("collection") or {}
    verification = run_report.get("verification") or {}
    lifecycle = run_report.get("lifecycle") or {}
    reasons, caveats = [], []

    for finding in collection.get("integrity") or []:
        if finding.get("level") == "blocking":
            reasons.append(f"integrity:{finding.get('code')}: {finding.get('detail')}")

    if not verification.get("passed"):
        failed = "; ".join(f"{f.get('code')}: {f.get('detail')}"
                           for f in verification.get("findings") or []
                           if f.get("level") == "fail")
        reasons.append("audit did not pass" + (f": {failed}" if failed else ""))

    reference = collection.get("reference") or {}
    decisive = {seq for seq in (reference.get("launch_seq"),
                                collection.get("best_launch_seq")) if seq is not None}
    for finding in verification.get("findings") or []:
        if finding.get("level") == "fail":
            continue
        # Deciding metric/witness contradictions are audit failures above. These
        # warnings instead identify missing evidence needed to recheck the result.
        # Lineage and grounded charge disclosures do not change its measurements.
        undermines_result = (
            finding.get("level") == "warn"
            and finding.get("launch_seq") in decisive
            and finding.get("code") in {"unverifiable_no_stdout", "unverifiable_source_metric"}
        )
        text = f"{finding.get('code')} at launch {finding.get('launch_seq')}: {finding.get('detail')}"
        (reasons if undermines_result else caveats).append(
            ("the result's own evidence is unverifiable: " if undermines_result else "") + text)

    if run_report.get("evidence_status") == "failed":
        reasons.append("evidence verification failed for this run")
    elif run_report.get("evidence_status") != "verified":
        caveats.append("Some launch evidence could not be verified; see the audit findings.")

    status = lifecycle.get("recorded_status")
    if status != "finished":
        reasons.append(f"the run is {status!r}; a finished run is required")

    if not reference.get("present"):
        reasons.append("no reference launch is recorded, so there is nothing to compare against")
    elif reference.get("mismatches"):
        reasons.append("the reference did not reproduce its pre-registered values: "
                       + "; ".join(str(m) for m in reference["mismatches"]))

    best = collection.get("best")
    if not best:
        reasons.append("no admissible candidate was measured")

    if not agrees:
        reasons.append("the selected candidate disagrees with the engine's admissibility verdict")

    return {"publishable": not reasons, "reasons": reasons, "caveats": caveats}


def _source_identity(run_dir: Path, candidate_id: object) -> dict | None:
    """Per-file digests for one candidate, as reserved at measurement time."""
    if not isinstance(candidate_id, str) or not candidate_id:
        return None
    return _read(run_dir / "engine/source-identities" / f"{candidate_id}.json") or None


def history(run, views, run_report: dict) -> list:
    """Every launch in order: the method's idea, the code's identity, and the result."""
    collection = run_report.get("collection") or {}
    run_dir = run.path
    by_uuid = {view.launch_uuid: view for view in views}
    out = []
    for launch in collection.get("launches") or []:
        uuid = launch.get("launch_uuid")
        view = by_uuid[uuid]
        intent, result = view.intent, view.result or {}
        logs = {}
        for name in ("stdout", "stderr"):
            recorded = result.get(name + "_path")
            path = runs.log_path(run, recorded) if recorded else None
            logs[name] = str(path.relative_to(run_dir)) if path and path.is_file() else None
        out.append({
            "launch_seq": launch.get("launch_seq"),
            "candidate_id": launch.get("candidate_id"),
            "parent_id": launch.get("parent_id"),
            "is_reference": bool(launch.get("is_reference_launch")),
            "status": launch.get("status"),
            "exit_code": launch.get("exit_code"),
            "witness": launch.get("witness"),
            "admissible": launch.get("admissible"),
            "clause": launch.get("clause"),
            "reason": launch.get("reason"),
            "charged": launch.get("charged"),
            "charge_classification": launch.get("charge_classification"),
            "charge_reason": launch.get("charge_reason"),
            "metrics": launch.get("all_metrics") or launch.get("metrics"),
            "gate_margin": launch.get("gate_margin"),
            "rank_key": launch.get("rank_key"),
            "extraction_errors": launch.get("extraction_errors"),
            # The method's own account of this proposal, verbatim and untrusted. It is a
            # claim about intent, never evidence about the measurement.
            "idea": intent.get("idea"),
            "research_log": intent.get("research_log"),
            "source": {"path": str(runs.candidate_source(run, launch["candidate_id"]).relative_to(run_dir)),
                       "archived_in_run": runs.candidate_source(run, launch["candidate_id"]).is_dir()},
            "source_identity": _source_identity(run_dir, launch.get("candidate_id")),
            "started_at": launch.get("started_at"),
            "finished_at": launch.get("finished_at"),
            "wall_time_seconds": launch.get("wall_time_seconds"),
            "lane": launch.get("lane"),
            **logs,
            "result_row_missing": view.result is None,
        })
    return out


def outcome(run_report: dict, state: dict, finished: dict) -> dict:
    """How the run ended, as a category plus the reason it recorded for itself."""
    lifecycle = run_report.get("lifecycle") or {}
    budget = (run_report.get("collection") or {}).get("budget") or {}
    status = lifecycle.get("recorded_status")
    reason = finished.get("reason")

    if status != "finished":
        category = "cut short"
    elif budget.get("remaining") == 0 or budget.get("overspent"):
        category = "budget exhausted"
    else:
        category = "the method stopped itself"

    anomalies = []
    if state.get("controller_outlived_completion"):
        anomalies.append("the controller process outlived the run's completion and was killed")
    exit_code = state.get("exit_code")
    if isinstance(exit_code, int) and exit_code not in (0, None):
        anomalies.append(f"the run process exited with code {exit_code}")

    return {"category": category, "recorded_status": status, "stop_reason": reason,
            "method_declared_complete": lifecycle.get("method_declared_complete"),
            "stop_requested": lifecycle.get("stop_requested"),
            "process_exit_code": exit_code, "anomalies": anomalies}


def setup(run_dir: Path, meta: dict, definition: dict) -> dict:
    """Public setup facts from the recorded launch configuration."""
    compute = meta.get("compute") or {}
    describe = compute.get("describe") or {}
    config = compute.get("config") or {}
    workers = meta.get("lanes")
    gpus = (meta.get("task_snapshot") or {}).get("launch", {}).get("gpus_per_launch")
    launch_config = definition.get("config") or {}
    bindings = launch_config.get("bindings") or {}
    declared_llm = launch_config.get("research_llm") or {}
    llm = {"model": declared_llm.get("model") or meta.get("research_model"),
           "provider": declared_llm.get("provider") or bindings.get("provider")}
    if "used" in declared_llm:
        llm["used"] = declared_llm["used"]
    settings = {key: bindings[key] for key in ("temperature", "top_p", "max_tokens", "reasoning_effort") if key in bindings}
    settings.update(declared_llm.get("settings") or {})
    if settings:
        llm["generation_settings"] = settings
    usage = []
    for log in sorted((run_dir / "engine").glob("attempt-*/stdout.log")):
        for line in log.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("type") == "result" and isinstance(event.get("usage"), dict):
                usage.append({"usage": event["usage"], "reported_cost_usd": event.get("total_cost_usd"),
                              "reported_turns": event.get("num_turns")})
                model_usage = event.get("modelUsage")
                providers = {
                    value["provider"] for value in model_usage.values()
                    if isinstance(value, dict) and isinstance(value.get("provider"), str)
                    and value["provider"]
                } if isinstance(model_usage, dict) else set()
                if len(providers) == 1 and not llm.get("provider"):
                    llm["provider"] = next(iter(providers))
    if usage:
        llm["usage"] = {"coverage": "Recorded controller responses; full-run coverage unverified.", "records": usage}
    return {
        "compute": {"backend": describe.get("backend") or compute.get("backend"),
                    "gpu_instance_type": config.get("job", {}).get("instance_type"),
                    "gpu_model": describe.get("gpu_names") or describe.get("accelerator"),
                    "workers": workers, "gpus_per_evaluation": gpus,
                    "evaluation_capacity_gpus": workers * gpus if workers and gpus else None},
        "research_llm": llm,
    }


def limits(definition: dict) -> list:
    """What this record cannot support, stated rather than left to be discovered."""
    out = []
    metric_defs = (definition or {}).get("metrics") or {}
    noisy = sorted(name for name, spec in metric_defs.items()
                   if spec.get("deterministic") is False)
    if noisy:
        out.append("Metrics the task declares non-deterministic: " + ", ".join(noisy) + ".")
    return out


def reconcile(run_dir: Path, record: dict, run_report: dict, views) -> dict:
    """Basic arithmetic re-derived from the ledger, so the record cannot contradict it."""
    collection = run_report.get("collection") or {}
    totals = collection.get("totals") or {}
    checks = []

    def check(name, expected, observed):
        checks.append({"check": name, "expected": expected, "observed": observed,
                       "agrees": expected == observed})

    check("launches recorded equals intent rows in the ledger",
          len(views), totals.get("launches_recorded"))

    counts = {}
    for view in views:
        counts[view.status] = counts.get(view.status, 0) + 1
    recorded = {k: v for k, v in (collection.get("status_counts") or {}).items() if v}
    checks.append({"check": "status counts recomputed from canonical launch views",
                   "expected": counts, "observed": recorded, "agrees": counts == recorded})

    wall = sum((view.result or {}).get("wall_time_seconds") or 0 for view in views)
    gpu = totals.get("gpu_seconds")
    checks.append({"check": "gpu seconds equal the sum of recorded wall times",
                   "expected": round(wall, 1), "observed": gpu,
                   "agrees": gpu is None or abs(round(wall, 1) - float(gpu)) <= 0.2})

    headline = record.get("headline") or {}
    best, reference = headline.get("best_value"), headline.get("reference_value")
    if isinstance(best, (int, float)) and isinstance(reference, (int, float)) and reference:
        direction = headline.get("direction")
        delta = (reference - best) if direction == "minimize" else (best - reference)
        checks.append({"check": "relative improvement recomputed from best and reference",
                       "expected": delta / abs(reference),
                       "observed": headline.get("relative_improvement"),
                       "agrees": headline.get("relative_improvement") is None
                       or abs(delta / abs(reference) - headline["relative_improvement"]) < 1e-9})

    for entry in record.get("evidence") or []:
        for name in ("stdout", "stderr"):
            relative = entry.get(name)
            if not relative or relative.startswith("/"):
                continue
            path = run_dir / relative
            size = path.stat().st_size if path.is_file() else None
            checks.append({"check": f"{entry.get('role')} {name} exists at its declared size",
                           "expected": entry.get(f"{name}_bytes"),
                           "observed": size,
                           "agrees": size is not None and size == entry.get(f"{name}_bytes")})

    return {"checks": checks, "disagreements": sum(1 for c in checks if not c["agrees"])}


def build(run_dir) -> dict:
    """Assemble one submission record. Reads only; writes nothing under `run_dir`."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "run.json").is_file():
        raise SubmissionError(f"{run_dir} is not a run directory (no run.json)")
    run_report, run, views, recorded = submission_data._run_report(run_dir)
    definition = run_report.get("task_definition") or {}
    collection = run_report.get("collection") or {}
    best = collection.get("best") or {}
    comparison = run_report.get("comparison") or {}

    winner_seq = collection.get("best_launch_seq")
    winning_launch = next((row for row in collection.get("launches", [])
                           if row["launch_seq"] == winner_seq), None)
    metrics = winning_launch["all_metrics"] if winning_launch else {}
    rows = eligibility(definition, metrics)
    view = _TaskView(definition)
    engine_verdict = None
    agrees = True
    if winning_launch:
        decided = objective_module.admissibility(view, metrics)
        engine_verdict = {"admissible": decided.admissible, "clause": decided.clause,
                          "reason": decided.reason}
        agrees = decided.admissible == winning_launch["admissible"]

    launches = history(run, views, run_report)
    reference_seq = (collection.get("reference") or {}).get("launch_seq")
    evidence = []
    for entry in launches:
        is_reference = entry["is_reference"] or entry["launch_seq"] == reference_seq
        is_winner = entry["launch_seq"] == winner_seq
        # A method that found nothing better leaves the reference as the best result, and
        # both roles land on one launch. Saying only "best" there loses the fact that the
        # winner is the baseline, which is the whole finding of such a run.
        role = ("reference and best" if is_reference and is_winner
                else "best" if is_winner else "reference" if is_reference else None)
        if role is None:
            continue
        item = {"role": role, "launch_seq": entry["launch_seq"],
                "candidate_id": entry["candidate_id"], "status": entry["status"],
                "stdout": entry["stdout"], "stderr": entry["stderr"],
                "source_identity": entry["source_identity"]}
        for name in ("stdout", "stderr"):
            relative = item[name]
            target = run_dir / relative if relative and not relative.startswith("/") else None
            item[f"{name}_bytes"] = target.stat().st_size if target and target.is_file() else None
        evidence.append(item)
    evidence.sort(key=lambda item: 0 if item["role"] == "reference" else 1)

    record = {
        "schema": SCHEMA,
        "verdict": verdict(run_report, agrees),
        "headline": {
            "task_id": definition.get("task_id"),
            "task_title": definition.get("title"),
            "method_id": (collection.get("method") or {}).get("method_id"),
            "metric": comparison.get("metric"),
            "unit": comparison.get("unit"),
            "direction": comparison.get("direction"),
            "best_value": comparison.get("best_value"),
            "reference_value": comparison.get("reference_value"),
            "absolute_improvement": comparison.get("absolute_improvement"),
            "relative_improvement": comparison.get("relative_improvement"),
            "strictly_better_than_reference": comparison.get("strictly_better_than_reference"),
            "reference_admissible": comparison.get("reference_admissible"),
            "best_is_reference": comparison.get("best_is_reference"),
            "best_launch_seq": winner_seq,
            "best_candidate_id": best.get("candidate_id"),
            "gate_margin": best.get("gate_margin"),
            "rank_key": best.get("rank_key"),
            "tiebreaks": view.objective_spec.get("tiebreaks"),
        },
        "eligibility": {"clauses": rows, "engine_verdict": engine_verdict,
                        "scope": "frozen task definition"},
        "obligations": obligations(definition),
        "history": launches,
        # Best-so-far and cumulative spend per charged launch, as the engine computed it.
        # Derivable from `history`, but recomputing it here would be a second opinion on
        # the ranking the engine already made.
        "trajectory": collection.get("trajectory"),
        "cost": {"budget": collection.get("budget"),
                 "status_counts": collection.get("status_counts"),
                 "totals": collection.get("totals"),
                 "status_meaning": {status: STATUS_MEANING[status]
                                    for status in (collection.get("status_counts") or {})
                                    if status in STATUS_MEANING}},
        "outcome": outcome(run_report, recorded["state"], recorded["finished"]),
        "setup": setup(run_dir, run.meta, recorded["definition"]),
        "evidence": evidence,
        "limits": limits(definition),
        "identity": {name: run.meta.get(name) for name in
                           ("run_id", "task_id", "task_version", "task_snapshot_sha256",
                            "method_id", "method_snapshot_sha256",
                            "engine_definition_sha256", "harness_revision",
                            "research_model", "created_at",
                            )},
        "audit": {"verification": run_report.get("verification"),
                  "integrity": collection.get("integrity"),
                  "evidence_status": run_report.get("evidence_status")},
        "task_definition": definition,
    }
    for entry in launches:
        if entry["stderr"] is None:
            record["verdict"]["caveats"].append(f"stderr is unavailable at launch {entry['launch_seq']}.")
    record["reconciliation"] = reconcile(run_dir, record, run_report, views)
    gpus = record["setup"]["compute"]["gpus_per_evaluation"]
    record["cost"]["evaluation_gpu_hours"] = (sum(
        (row.get("wall_time_seconds") or 0) * gpus for row in launches) / 3600 if gpus else None)
    record["cost"]["gpu_time_coverage"] = "Evaluations with recorded durations; excludes missing durations."
    changes = []
    record = submission_data._strict_json(record, changes)
    record["nonfinite_values"] = changes
    return record
