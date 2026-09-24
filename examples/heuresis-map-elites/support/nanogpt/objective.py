"""Apply the supplied task's gated objective to Heuresis feedback.

Canonical metrics are checked against the delivered stdout before archive
admission. Fitness combines the target with a bounded tiebreak fraction.
The engine's objective rules determine eligibility.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------

_CACHE: dict[str, Any] = {}


def _document() -> dict:
    if "document" not in _CACHE:
        raw = os.environ.get("AUTOARENA_CONTEXT")
        if not raw:
            raise RuntimeError("AUTOARENA_CONTEXT is required")
        _CACHE["document"] = json.loads(Path(raw).read_text())
    return _CACHE["document"]


def repo_root() -> Path:
    return Path(_document()["runner_root"])


def run_dir() -> Path:
    return Path(_document()["run_dir"])


def _autoarena():
    """Reuse runner rules on the host; never expose the runner inside the sandbox."""
    if "autoarena" not in _CACHE:
        sys.path.insert(0, str(repo_root()))
        from engine.evaluation import task as arena_task, objective as arena_objective
        _CACHE["autoarena"] = (arena_task, arena_objective)
    return _CACHE["autoarena"]


def task() -> Any:
    if "task" not in _CACHE:
        arena_task, _ = _autoarena()
        document = _document()
        path = Path(document["task_dir"]) / "task.json"
        raw = json.loads(path.read_text())
        arena_task.validate(raw, path)
        if raw != document["task"]:
            raise ValueError("Frozen task file differs from the engine context")
        # Frozen packages live at engine/task, not tasks/<task_id>. Preserve the
        # validated recorded bytes without the registry loader's directory rule.
        _CACHE["task"] = arena_task.Task(
            task_id=raw["task_id"], version=raw["version"], title=raw["title"],
            path=path, raw=raw,
        )
    return _CACHE["task"]


# ---------------------------------------------------------------------------
# Task objective
# ---------------------------------------------------------------------------


def load_ranking(the_task: Any | None = None) -> list[dict]:
    """Require a gated, minimizing target and at most one minimizing tiebreak."""
    the_task = the_task or task()
    objective = the_task.objective_spec
    where = f"{the_task.task_id}/task.json"
    if the_task.comparison_mode != "gated":
        raise ValueError(f"{where}: this module implements only gated comparison")
    gate = objective.get("quality_gate")
    if not isinstance(gate, dict) or not {"metric", "operator", "value"} <= gate.keys():
        raise ValueError(f"{where}: a quality_gate is required")
    target = objective.get("target")
    if not isinstance(target, dict) or target.get("direction") != "minimize":
        raise ValueError(f"{where}: this module requires a minimizing target")
    ties = objective.get("tiebreaks")
    if not isinstance(ties, list) or len(ties) > 1:
        raise ValueError(f"{where}: this scalar implements at most one tiebreak")
    if any(entry.get("direction") != "minimize" for entry in ties):
        raise ValueError(f"{where}: this module requires a minimizing tiebreak")
    return [target, *ties]


def ranking_metric() -> str:
    return load_ranking()[0]["metric"]


def tiebreak_metric() -> str | None:
    secondary = load_ranking()
    return secondary[1]["metric"] if len(secondary) > 1 else None


def gate_metric() -> str:
    return task().objective_spec["quality_gate"]["metric"]


def gate_bound() -> tuple[float, str]:
    gate = task().objective_spec["quality_gate"]
    return float(gate["value"]), str(gate["operator"])


def ceiling_metrics() -> list[str]:
    return [c["metric"] for c in task().objective_spec.get("constraints", [])]


def objective_label() -> str:
    value, operator = gate_bound()
    tie = tiebreak_metric()
    tail = f", tiebreak {tie} minimize" if tie else ""
    ceilings = ", ".join(
        f"{c['metric']} {c['operator']} {c['value']}"
        for c in task().objective_spec.get("constraints", [])
    )
    return (
        f"{gate_metric()} {operator} {value} (GATE ONLY); "
        f"{ranking_metric()} minimize (ranking key){tail}"
        + (f"; ceilings: {ceilings}" if ceilings else "")
    )


# ---------------------------------------------------------------------------
# reading the launch -- METRICS_JSON is the accessor's only source
# ---------------------------------------------------------------------------

# The task's immutable instrument prints canonical metrics on this line.
_METRICS_JSON_RE = re.compile(r"^METRICS_JSON:\s*(\{.*\})\s*$", re.MULTILINE)


def parse_metrics_json(run_log_text: str) -> dict[str, Any] | None:
    """The last ``METRICS_JSON: {...}`` object in a launch log, at full precision.

    An absent or malformed object returns None.
    """
    matches = _METRICS_JSON_RE.findall(run_log_text or "")
    if not matches:
        return None
    try:
        parsed = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def read_ranking_value(run_log_text: str) -> float | None:
    """This axis's ranking key, by name, out of the one metric line."""
    parsed = parse_metrics_json(run_log_text)
    if parsed is None:
        return None
    value = parsed.get(ranking_metric())
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# scalar fitness for a two-level ranking key
# ---------------------------------------------------------------------------

# Archive fitness adds a tiebreak fraction below 0.5 to the raw target.
# Omit the fraction when it is smaller than the configured multiple of the
# target's floating-point spacing. See support/README.md for ordering limits.
_FRACTION_CEILING = 0.5
_MIN_ULP_MULTIPLE = 1024.0


def composite_fitness(rank_value: float, tiebreak_value: float | None) -> float:
    """Combine the raw target with a bounded fraction of the tiebreak."""
    rank = float(rank_value)
    if tiebreak_value is None:
        return rank
    import math  # noqa: PLC0415

    tie = max(float(tiebreak_value), 0.0)
    # Normalise the tiebreak into [0, 1) against its own order of magnitude, so the
    # encoding does not depend on which metric the task nominates.
    decades = math.floor(math.log10(tie)) + 1 if tie > 0 else 1
    normalised = min(tie / (10.0 ** max(decades, 1)), 0.999999)
    fraction = normalised * _FRACTION_CEILING
    # Refuse to encode a tiebreak the float cannot represent beside this ranking value.
    # Dropping it is correct and reporting it is necessary: a tiebreak silently rounded
    # to zero would order an exact tie arbitrarily while looking like it had a rule.
    ulp = math.ulp(rank) if rank else math.ulp(1.0)
    if fraction < ulp * _MIN_ULP_MULTIPLE:
        return rank
    return rank + fraction


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    """Outcome of applying this axis's gated objective to one launch."""

    admissible: bool
    fitness: float | None
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def status(self) -> str:
        if self.admissible:
            return "ok"
        return self.reason or "reject"


# The queue bridge delivers this to the executor after Benchmark.evaluate. It is
# the ledger row's own metric dict, so a candidate cannot be scored on anything the
# ledger does not also say.
RESULT_FILENAME = "arena_result.json"
RUN_LOG_FILENAME = "run.log"

# Metric names and units consumed by the native archive records.
_RECORD_ALIASES = {
    "peak_vram_mb": ("peak_vram_bytes", 1.0 / (1024.0 * 1024.0)),
    "num_params": ("num_params_total", 1.0),
    "total_tokens_M": ("total_tokens", 1e-6),
}

def evaluate(info: dict, workspace: Any, **_ignored: Any) -> Verdict:
    """Apply this axis's contract to one finished launch.

    ``info`` is the pinned harness's grader output and is deliberately NOT the source of
    any number here: the numbers come from the ledger-backed ``arena_result.json`` and
    are cross-checked against the launch's own ``METRICS_JSON`` line.
    """
    the_task = task()
    _, arena_objective = _autoarena()
    load_ranking(the_task)  # the guard, on every launch, not once at startup

    ws = Path(workspace) if workspace is not None else None
    metrics: dict[str, Any] = {}

    if ws is None:
        return Verdict(False, None, metrics, reason="no_workspace")

    # The delivered result is read straight out of the workspace, beside the log it
    # belongs to: `dispatch.py` writes both in the same hand-back, so the numbers scored
    # here are the ones the graded log carries. Nothing is re-derived from the queue and
    # nothing about the workspace is adjudicated -- the engine bound source bytes to
    # metrics when it charged the launch, and the ledger, not this file, is the record.
    result_path = ws / RESULT_FILENAME
    if not result_path.is_file():
        # The executor never dispatched a launch, so nothing was measured and nothing was
        # charged. Recorded as its own reason rather than folded into a crash: an agent
        # that failed to reach the GPU at all is a different event from a candidate that
        # reached it and died, and the two have different accounting.
        return Verdict(False, None, metrics, reason="never_dispatched")

    try:
        record = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return Verdict(False, None, metrics, reason=f"unreadable_result:{error}")

    metrics = dict(record.get("metrics") or {})
    metrics["launch_seq"] = record.get("launch_seq")
    metrics["candidate_id"] = record.get("candidate_id")
    metrics["charge"] = record.get("charge")

    for alias, (source, scale) in _RECORD_ALIASES.items():
        value = metrics.get(source)
        if value is not None and alias not in metrics:
            try:
                metrics[alias] = float(value) * scale
            except (TypeError, ValueError):
                pass

    status = str(record.get("status") or "")
    if status != "ok":
        # The harness already classified this from the row's own account of what
        # happened. Not re-derived here, and in particular not re-derived from duration:
        # a node outage and a candidate OOM are both fast and only the second is a
        # property of the candidate.
        return Verdict(False, None, metrics, reason=status or "unknown_status")

    # Independent re-reading of the launch's own metric line, and a refusal on
    # disagreement. See the module docstring: this is the check against scoring a stale
    # log or another launch's result file as this candidate's measurement.
    log_path = ws / "regenerated" / RUN_LOG_FILENAME if info.get("regenerated") else ws / RUN_LOG_FILENAME
    printed = parse_metrics_json(log_path.read_text(errors="replace")
                                if log_path.is_file() else "")
    if printed is None:
        return Verdict(False, None, metrics, reason="no_metrics_json_in_run_log")

    checked = [ranking_metric(), gate_metric(), *ceiling_metrics()]
    tie = tiebreak_metric()
    if tie:
        checked.append(tie)
    for name in dict.fromkeys(checked):
        left, right = metrics.get(name), printed.get(name)
        if left is None or right is None:
            return Verdict(False, None, metrics,
                           reason=f"metric_missing:{name}")
        try:
            agree = float(left) == float(right)
        except (TypeError, ValueError):
            agree = False
        if not agree:
            return Verdict(
                False, None, metrics,
                reason=f"metrics_disagree:{name}:{left}!={right}")

    # Native judging can invalidate numerically eligible candidates. Preserve its
    # fail-closed decision and its regrading result before archive admission.
    if "judge_verdict" in info and (not info.get("valid") or info.get("best_score") is None):
        return Verdict(False, None, metrics, reason="judge_rejected")

    verdict = arena_objective.admissibility(the_task, metrics)
    if not verdict.admissible:
        return Verdict(False, None, metrics,
                       reason=f"inadmissible:{verdict.clause}")

    rank_value = metrics.get(ranking_metric())
    if rank_value is None:
        return Verdict(False, None, metrics, reason="no_ranking_value")

    tie_value = metrics.get(tie) if tie else None
    metrics["gate_margin"] = arena_objective.gate_margin(the_task, metrics)
    return Verdict(True,
                   composite_fitness(float(rank_value), tie_value),
                   metrics, reason="admissible")
