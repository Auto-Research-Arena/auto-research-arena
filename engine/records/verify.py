"""Check one ledger snapshot against launch logs, captured source and task rules.

Missing evidence produces warnings; `passed` means no failures were found, not
that every measurement could be verified. Checked counts record the coverage.
"""

from __future__ import annotations

from engine.records.runs import log_path

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.evaluation.task import Task
from engine.records.charging import charge as charge_rule
from engine.evaluation.metrics import extract_metrics
from engine.evaluation.objective import admissibility

from engine.records.collect import summary_metrics
from engine.records.ledger import LaunchView
from engine.records.runs import Run

# How close a re-extracted float must be to the recorded one. Not zero: JSON round-trips a
# float exactly, but a metric derived as a product of two others can differ in the last
# bit depending on the order the factors were multiplied in, and failing an audit on that
# would train readers to ignore the audit.
FLOAT_TOLERANCE = 1e-9


@dataclass
class Finding:
    level: str  # "fail" | "warn" | "note"
    code: str
    detail: str
    launch_seq: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Report:
    run_id: str
    task_id: str
    method_id: str
    findings: List[Finding] = field(default_factory=list)
    checked: Dict[str, int] = field(default_factory=dict)

    @property
    def failures(self) -> List[Finding]:
        return [item for item in self.findings if item.level == "fail"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def add(self, level: str, code: str, detail: str, launch_seq: Optional[int] = None) -> None:
        self.findings.append(Finding(level, code, detail, launch_seq))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "autoarena/verification/1",
            "run_id": self.run_id,
            "task_id": self.task_id,
            "method_id": self.method_id,
            "passed": self.passed,
            "checked": self.checked,
            "findings": [item.to_dict() for item in self.findings],
        }


def verify(
    run: Run,
    *,
    task: Optional[Task] = None,
    source_roots: Optional[Dict[str, Path]] = None,
    views: Optional[List[LaunchView]] = None,
) -> Report:
    """Recompute the run and report every disagreement.

    `source_roots` maps candidate_id to the candidate's own substrate copy, needed to
    re-extract `source_constant` metrics. Passing `None` discovers the engine's captured
    copies under `engine/source-snapshots/`; passing
    an explicit mapping (including an empty one) uses exactly that. Whatever is missing is
    reported as unverifiable rather than assumed correct: a constant lives in a mutable
    file, so reading it from the pristine substrate would verify the wrong source and pass.

    `views` may supply an already validated canonical snapshot, including [].
    Only None reads and folds the run's ledger.
    """
    task = task or run.task()
    report = Report(run_id=run.run_id, task_id=task.task_id, method_id=run.method_id)
    if views is None:
        views = run.views()  # raises on a malformed ledger
    report.checked["launches"] = len(views)

    deciding = set(summary_metrics(task))
    needs_source_metrics = source_dependent(task)
    verified_logs = 0
    if source_roots is None:
        source_roots = discover_source_roots(run)

    for view in views:
        seq = view.launch_seq
        if not view.resolved:
            report.add(
                "fail",
                "unresolved_launch",
                "an intent row with no result. It is charged, so the budget is right, but "
                "nothing records what the GPU time bought",
                seq,
            )
            continue

        result = view.result or {}
        stdout_path = result.get("stdout_path")
        if not stdout_path or not log_path(run, stdout_path).is_file():
            report.add(
                "warn",
                "unverifiable_no_stdout",
                f"stdout_path {stdout_path!r} is missing, so the witness and every metric "
                "on this row are taken on trust",
                seq,
            )
            continue
        verified_logs += 1
        stdout = log_path(run, stdout_path).read_text(encoding="utf-8", errors="replace")

        # -- the witness, from this launch's own output --------------------
        observed_witness = task.gpu_work_witness in stdout
        claimed_witness = bool(result.get("witness"))
        if observed_witness != claimed_witness:
            report.add(
                "fail",
                "witness_mismatch",
                f"the row claims witness={claimed_witness} and this launch's stdout says "
                f"{observed_witness}. The witness decides charging, so a false positive "
                "bills the method for our plumbing and a false negative gives back a "
                "launch that consumed a GPU",
                seq,
            )

        # -- the metrics, re-extracted -------------------------------------
        source_root = (source_roots or {}).get(view.candidate_id)
        observed, errors = extract_metrics(task, stdout, source_root=source_root)
        recorded = view.metrics
        for metric in sorted(set(observed) | set(recorded)):
            if metric in needs_source_metrics and source_root is None:
                if metric in deciding:
                    report.add(
                        "warn",
                        "unverifiable_source_metric",
                        f"{metric} is read from the candidate's source and no source copy "
                        "was given, so it cannot be rechecked. It decides admissibility "
                        "or ranking",
                        seq,
                    )
                continue
            if not _same(observed.get(metric), recorded.get(metric)):
                report.add(
                    "fail" if metric in deciding else "warn",
                    "metric_mismatch",
                    f"{metric}: the ledger records {recorded.get(metric)!r} and "
                    f"re-extraction from stdout gives {observed.get(metric)!r}"
                    + (
                        ". This metric ranks or admits candidates, so the row is not "
                        "comparable"
                        if metric in deciding
                        else ""
                    ),
                    seq,
                )
        for error in errors:
            if source_root is None and any(
                error.startswith(name + ":") for name in needs_source_metrics
            ):
                continue  # an artefact of the evidence we were not given, already reported
            if error not in (result.get("extraction_errors") or []):
                report.add(
                    "warn",
                    "undeclared_extraction_error",
                    f"re-extraction reports {error!r}, which the row does not list",
                    seq,
                )

        # -- admissibility, re-decided -------------------------------------
        verdict = admissibility(task, recorded)
        report.checked.setdefault("admissibility", 0)
        report.checked["admissibility"] += 1
        if verdict.admissible and _not_finite_any(recorded, deciding):
            report.add(
                "fail",
                "admissible_on_unusable_numbers",
                "the recorded metrics pass admissibility but a deciding metric is not a "
                "finite number",
                seq,
            )

        # -- charging, re-derived ------------------------------------------
        rule = charge_rule(result)
        folded = view.charge()
        if view.adjudications and folded.decision != rule.decision:
            report.add(
                "note",
                "adjudicated_against_the_rule",
                f"the rule says {rule.decision} and an adjudication says "
                f"{folded.decision}: {folded.reason}",
                seq,
            )

    report.checked["stdout_reread"] = verified_logs

    _verify_budget(views, task, report)
    _verify_reference(views, task, report)
    _verify_lineage(views, report)
    _verify_research_logs(views, report)
    return report


# ---------------------------------------------------------------------------
# run-level checks
# ---------------------------------------------------------------------------


def _verify_budget(views: List[LaunchView], task: Task, report: Report) -> None:
    """Re-derive the spend. No stored counter is consulted, because none exists."""
    charged = {view.launch_uuid for view in views if view.charge().charged}
    report.checked["charged"] = len(charged)
    if len(charged) > task.max_launches:
        report.add(
            "fail",
            "budget_overspent",
            f"{len(charged)} charged launches against a cap of {task.max_launches}. The "
            "best-so-far is the best over a larger search than every other row",
        )
    if task.budget.get("includes_reference_launch"):
        reference = [view for view in views if view.is_reference_launch]
        if reference and not any(view.charge().charged for view in reference):
            report.add(
                "warn",
                "reference_not_charged",
                "the task's budget includes the reference launch and no reference launch is "
                "charged, so this run had one more candidate launch than the cap allows",
            )


def _verify_reference(views: List[LaunchView], task: Task, report: Report) -> None:
    reference = [view for view in views if view.is_reference_launch]
    if not reference:
        report.add(
            "fail",
            "no_reference_launch",
            "no launch is marked is_reference_launch, so nothing establishes that this "
            "machine reproduces the substrate. Every candidate number inherits the "
            "discrepancy that was never measured",
        )
        return
    metrics = reference[0].metrics
    for metric, want in (task.reference_launch.get("assert_exact") or {}).items():
        got = metrics.get(metric)
        if got is None or float(got) != float(want):
            report.add(
                "fail",
                "reference_mismatch",
                f"the reference launch recorded {metric}={got!r} and the task asserts "
                f"{want!r}. This quantity is deterministic given the source, so it is a "
                "different substrate rather than run-to-run variation",
                reference[0].launch_seq,
            )
    report.checked["reference_launches"] = len(reference)


def _verify_lineage(views: List[LaunchView], report: Report) -> None:
    """Check recorded parentage; parent selection and acceptance belong to the method."""
    candidates = {view.candidate_id: view for view in views}
    checked = 0
    for view in views:
        parent = view.parent_id
        if not parent:
            continue
        checked += 1
        if parent not in candidates:
            report.add(
                "warn",
                "unknown_parent",
                f"parent_id {parent!r} has no launch in this run; lineage cannot be replayed",
                view.launch_seq,
            )
            continue
        if candidates[parent].launch_seq >= view.launch_seq:
            report.add(
                "warn",
                "noncausal_parent",
                f"parent {parent!r} is at seq {candidates[parent].launch_seq}, not before its child",
                view.launch_seq,
            )
    report.checked["parent_links"] = checked


def _verify_research_logs(views: List[LaunchView], report: Report) -> None:
    """Check the recorded payload contract, without judging proposal content."""
    from engine.evaluation.protocol import ProtocolError, research_payload

    checked = 0
    for view in views:
        if view.is_reference_launch:
            continue
        payload = view.intent.get("research_log")
        try:
            research_payload(payload)
        except ProtocolError as error:
            report.add("fail", "invalid_research_log", str(error), view.launch_seq)
        else:
            checked += 1
    report.checked["research_logs"] = checked


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def source_dependent(task: Task) -> set:
    """Source-extracted metrics and every product that depends on them."""
    direct = set()
    for name, spec in task.metrics.items():
        if spec.get("extract", {}).get("kind") in ("source_constant", "source_regex"):
            direct.add(name)
    # Fixed point over the product graph. Small and shallow, so iterate rather than recurse.
    for _ in range(len(task.metrics) + 1):
        grown = set(direct)
        for name, spec in task.metrics.items():
            extract = spec.get("extract", {})
            if extract.get("kind") == "product" and any(
                factor in direct for factor in extract.get("factors", ())
            ):
                grown.add(name)
        if grown == direct:
            break
        direct = grown
    return direct


def discover_source_roots(run: Run) -> Dict[str, Path]:
    """Read the engine's canonical measurement snapshots."""
    root = run.path / "engine/source-snapshots"
    return {path.name: path for path in sorted(root.iterdir()) if path.is_dir()} if root.is_dir() else {}


def _same(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return abs(float(left) - float(right)) <= FLOAT_TOLERANCE * max(
            1.0, abs(float(left)), abs(float(right))
        )
    except (TypeError, ValueError):
        return left == right


def _not_finite_any(metrics: Dict[str, Any], names: Any) -> bool:
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            if not math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            return True
    return False
