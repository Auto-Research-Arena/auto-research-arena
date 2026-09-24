"""Load and validate an AutoArena task definition.

`schema/task.schema.json` documents the shape. This module is the authority,
because the rules that matter here are semantic and a JSON Schema cannot state
them: an objective metric must be declared in `metrics`; a ceiling may not name
the metric it is supposed to be bounding the side effects of; a reference launch
may only assert quantities that are deterministic given the code.

Validation fails closed. A task that is missing a field, or that carries a
placeholder, is a blocking error rather than a default -- the alternative is a
benchmark that silently invents a target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.autoarena import objective_spec
from engine.compute.probe import matches
from engine import ROOT as REPO_ROOT

TASKS_ROOT = REPO_ROOT / "tasks"

OPERATORS = ("<", "<=", ">", ">=", "==")
#: `==` is not a ceiling in the ordinary sense: it pins a quantity to the value the
#: reference launch reported, so an axis cannot be bought by moving it. Only a
#: deterministic metric may carry it -- see `_objective`.
EQUALITY_OPERATORS = ("==",)
EXTRACT_KINDS = (
    "summary_field",
    "block_field",
    "metrics_json",
    "source_constant",
    "source_regex",
    "product",
)
# A pattern match over candidate source is a heuristic, so it may record a fact but
# never decide one. Nothing extracted this way may be an objective or a ceiling.
ADVISORY_EXTRACT_KINDS = ("source_regex",)
PLACEHOLDER_MARKERS = ("replace-me", "replace-with", "TODO", "FIXME", "<fill", "xxx")

# A metric marked `deterministic` is fixed by the candidate's source rather than
# by the run, so a reference launch may assert it exactly and needs no tolerance.
# Asserting anything else exactly would require a dispersion estimate, which is a
# statistical test this benchmark does not run. The flag is a claim about the
# quantity, not about its extractor: `depth` is printed by the summary block and
# is still deterministic, while `training_seconds` is printed the same way and is not.


class TaskError(Exception):
    """A task definition is unusable. Never recovered from by guessing."""


@dataclass(frozen=True)
class Task:
    task_id: str
    version: int
    title: str
    path: Path
    raw: Dict[str, Any]

    # -- convenience accessors, so callers never re-walk the dict ------------
    @property
    def substrate(self) -> Dict[str, Any]:
        return self.raw["substrate"]

    @property
    def source_root(self) -> Path:
        """The packaged code beside the task definition."""
        return self.path.parent / "code"

    @property
    def launch(self) -> Dict[str, Any]:
        return self.raw["launch"]

    @property
    def budget(self) -> Dict[str, Any]:
        return self.raw["budget"]

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.raw["metrics"]

    @property
    def objective(self) -> Dict[str, Any]:
        """The task's explicit objective definition."""
        return self.raw["objective"]

    @property
    def objective_spec(self) -> Dict[str, Any]:
        """Explicit read-only view; never used to serialize or identify the task."""
        return objective_spec(self.objective)

    @property
    def reference_launch(self) -> Dict[str, Any]:
        return self.raw["reference_launch"]

    @property
    def report(self) -> Dict[str, Any]:
        return self.raw["report"]

    @property
    def comparison_mode(self) -> str:
        return self.objective_spec["comparison_mode"]

    @property
    def max_launches(self) -> int:
        return int(self.budget["max_launches"])

    @property
    def gpu_work_witness(self) -> str:
        return self.substrate["gpu_work_witness"]

    @property
    def headline_metric(self) -> str:
        return self.report["headline"]

    @property
    def headline_direction(self) -> str:
        return self.report.get("headline_direction", "minimize")

    @property
    def ranking_metrics(self) -> List[Dict[str, Any]]:
        """Optimization target followed by the declared tiebreaks."""
        spec = self.objective_spec
        return [spec["target"], *spec["tiebreaks"]]


def accelerator_error(task: Task, gpu_ids: List[str], names: List[str]) -> Optional[str]:
    """Check observed models for the selected devices against the task contract."""
    if not gpu_ids or len(names) != len(gpu_ids):
        return "cannot observe the selected GPUs: " + ", ".join(gpu_ids)
    required = task.launch["accelerator"]
    if not all(matches(required, name) for name in names):
        return f"task requires {required}; selected GPUs: " + ", ".join(
            f"{device}: {name}" for device, name in zip(gpu_ids, names)
        )
    return None


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def available_tasks(root: Optional[Path] = None) -> List[str]:
    base = Path(root) if root else TASKS_ROOT
    if not base.is_dir():
        return []
    return sorted(p.parent.name for p in base.glob("*/task.json"))


def load_task(task_id: str, root: Optional[Path] = None) -> Task:
    base = Path(root) if root else TASKS_ROOT
    path = base / task_id / "task.json"
    if not path.is_file():
        known = ", ".join(available_tasks(base)) or "none"
        raise TaskError(f"no task {task_id!r} under {base} (known: {known})")
    return load_task_file(path)


def load_task_file(path: Path) -> Task:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise TaskError(f"{path}: not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise TaskError(f"{path}: expected a JSON object")
    validate(raw, path)
    if raw["task_id"] != path.parent.name:
        raise TaskError(
            f"{path}: task_id {raw['task_id']!r} does not match its directory "
            f"{path.parent.name!r}; the id is how every artifact refers to it"
        )
    return Task(
        task_id=raw["task_id"],
        version=int(raw["version"]),
        title=raw["title"],
        path=path,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

REQUIRED_TOP = (
    "task_id",
    "version",
    "title",
    "substrate",
    "launch",
    "budget",
    "metrics",
    "objective",
    "reference_launch",
    "report",
)


def validate(raw: Dict[str, Any], path: Path = Path("<memory>")) -> None:
    errors: List[str] = []
    _structure(raw, errors)
    if errors:  # later checks assume the structure holds
        raise TaskError(_render(path, errors))
    _metrics(raw, errors)
    _objective(raw, errors)
    if not errors:
        _reference(raw, errors)
    _report(raw, errors)
    _placeholders(raw, errors)
    if errors:
        raise TaskError(_render(path, errors))


def _render(path: Path, errors: List[str]) -> str:
    body = "\n".join(f"  - {error}" for error in errors)
    return f"{path}: {len(errors)} problem(s) in the task definition\n{body}"


def _structure(raw: Dict[str, Any], errors: List[str]) -> None:
    for field in REQUIRED_TOP:
        if field not in raw:
            errors.append(f"missing required field {field!r}")
    if errors:
        return
    for field in ("substrate", "launch", "budget", "reference_launch", "report"):
        if not isinstance(raw[field], dict):
            errors.append(f"{field} must be an object")
    if errors:
        return
    substrate = raw["substrate"]
    required_substrate = (
        "id",
        "source",
        "revision",
        "prepare",
        "entrypoint",
        "mutable",
        "immutable",
        "gpu_work_witness",
    )
    for field in sorted(set(substrate) - set(required_substrate) - {"frozen_regions", "known_constraints"}):
        errors.append(f"substrate.{field} is not a substrate field")
    for field in required_substrate:
        if field not in substrate:
            errors.append(f"substrate: missing {field!r}")
    if isinstance(substrate.get("mutable"), list) and not substrate["mutable"]:
        errors.append("substrate.mutable is empty: nothing is editable, so no method can act")
    overlap = set(substrate.get("mutable") or []) & set(substrate.get("immutable") or [])
    if overlap:
        errors.append(
            "substrate: " + ", ".join(sorted(overlap)) + " listed both mutable and "
            "immutable; on conflict immutable wins, so remove it from mutable rather "
            "than leaving the contradiction for a reader to resolve"
        )
    if not str(substrate.get("gpu_work_witness") or "").strip():
        errors.append(
            "substrate.gpu_work_witness is empty: charging is grounded in that string "
            "appearing in a launch's own stdout, so without it the budget becomes a judgement"
        )

    # Validate declarations, not candidate behavior. These are semantic instructions
    # for cooperative participants and source audits, not a Python security policy.
    constraints = substrate.get("known_constraints", [])
    if not isinstance(constraints, list) or any(
        not isinstance(rule, str) or not rule.strip() for rule in constraints
    ):
        errors.append("substrate.known_constraints must be a list of non-empty strings")
    regions = substrate.get("frozen_regions", [])
    if not isinstance(regions, list):
        errors.append("substrate.frozen_regions must be a list")
    else:
        for index, region in enumerate(regions):
            label = f"substrate.frozen_regions[{index}]"
            if not isinstance(region, dict):
                errors.append(f"{label} must be an object")
                continue
            for field in ("file", "region", "reason"):
                if not isinstance(region.get(field), str) or not region[field].strip():
                    errors.append(f"{label}.{field} must be a non-empty string")
            if "anchor" in region and (
                not isinstance(region["anchor"], str) or not region["anchor"].strip()
            ):
                errors.append(f"{label}.anchor must be a non-empty string when present")

    launch = raw["launch"]
    for field, minimum in (
        ("train_seconds", 0),
        ("gpus_per_launch", 1),
        ("timeout_seconds", 0),
    ):
        value = launch.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum or value == 0:
            errors.append(f"launch.{field} must be a positive number")
    if "seed" not in launch:
        errors.append("launch.seed must be explicit, including 0")
    if not str(launch.get("accelerator") or "").strip():
        errors.append(
            "launch.accelerator must name the device the reference was measured on: "
            "a fixed wall-clock budget makes results incomparable across accelerators"
        )
    if (
        isinstance(launch.get("timeout_seconds"), (int, float))
        and isinstance(launch.get("train_seconds"), (int, float))
        and launch["timeout_seconds"] <= launch["train_seconds"]
    ):
        errors.append(
            "launch.timeout_seconds must exceed train_seconds: the budget excludes "
            "startup and compilation, so a timeout at the training budget kills every launch"
        )

    budget = raw["budget"]
    if not isinstance(budget.get("max_launches"), int) or budget.get("max_launches", 0) < 1:
        errors.append("budget.max_launches must be a positive integer")
    for field in ("includes_failures", "includes_reference_launch"):
        if not isinstance(budget.get(field), bool):
            errors.append(f"budget.{field} must be an explicit boolean")
    if budget.get("charging_rule") != "gpu-work-witness":
        errors.append(
            "budget.charging_rule must be 'gpu-work-witness': it is the only rule "
            "implemented, and a task naming another one would be silently ignored"
        )
    if budget.get("includes_failures") is False:
        errors.append(
            "budget.includes_failures must be true: a method that is not charged for "
            "failures can buy unlimited information with crashed launches"
        )


def _metrics(raw: Dict[str, Any], errors: List[str]) -> None:
    metrics = raw["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        errors.append("metrics must be a non-empty object")
        return
    for name, spec in metrics.items():
        if not isinstance(spec, dict):
            errors.append(f"metrics.{name} must be an object")
            continue
        if spec.get("type") not in ("integer", "number", "boolean"):
            errors.append(f"metrics.{name}.type must be integer, number, or boolean")
        extract = spec.get("extract")
        if not isinstance(extract, dict):
            errors.append(f"metrics.{name}.extract is required")
            continue
        kind = extract.get("kind")
        if kind not in EXTRACT_KINDS:
            errors.append(
                f"metrics.{name}.extract.kind {kind!r} is not one of {', '.join(EXTRACT_KINDS)}"
            )
            continue
        needed = {
            "metrics_json": ("field",),
            "summary_field": ("field",),
            "block_field": ("block", "field"),
            "source_constant": ("file", "name"),
            "source_regex": ("file", "pattern"),
            "product": ("factors",),
        }[kind]
        for field in needed:
            if field not in extract:
                errors.append(f"metrics.{name}.extract requires {field!r} for kind {kind!r}")
        if kind == "product":
            factors = extract.get("factors") or []
            if len(factors) < 2:
                errors.append(f"metrics.{name}: a product needs at least two factors")
            for factor in factors:
                if factor not in metrics:
                    errors.append(
                        f"metrics.{name}: factor {factor!r} is not itself a declared metric"
                    )
        cross = spec.get("cross_check")
        if isinstance(cross, dict) and cross.get("against") not in metrics:
            errors.append(
                f"metrics.{name}.cross_check.against {cross.get('against')!r} is not a declared metric"
            )


def _objective(raw: Dict[str, Any], errors: List[str]) -> None:
    objective = raw["objective"]
    if not isinstance(objective, dict):
        errors.append("objective must be an object")
        return
    start = len(errors)
    fields = {"comparison_mode", "target", "quality_gate", "tiebreaks", "constraints"}
    for key in sorted(set(objective) - fields):
        errors.append(f"objective.{key} is not an objective field")
    if objective.get("comparison_mode") != "gated":
        errors.append("objective.comparison_mode must be gated")
    for key in ("target", "quality_gate"):
        entry = objective.get(key)
        required = {"metric", "direction"} if key == "target" else {"metric", "operator", "value"}
        if not isinstance(entry, dict) or set(entry) != required:
            errors.append(f"objective.{key} must contain exactly {', '.join(sorted(required))}")
    for key in ("tiebreaks", "constraints"):
        entries = objective.get(key)
        if not isinstance(entries, list):
            errors.append(f"objective.{key} must be a list, explicitly empty if there are none")
            continue
        required = {"metric", "direction"} if key == "tiebreaks" else {"metric", "operator", "value", "basis"}
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != required:
                errors.append(f"objective.{key}[{index}] must contain exactly {', '.join(sorted(required))}")
    if len(errors) != start:
        return

    metrics = raw["metrics"] if isinstance(raw.get("metrics"), dict) else {}
    ranking = [("target", objective["target"])] + [
        (f"tiebreaks[{index}]", entry) for index, entry in enumerate(objective["tiebreaks"])
    ]
    bounds = [("quality_gate", objective["quality_gate"])] + [
        (f"constraints[{index}]", entry) for index, entry in enumerate(objective["constraints"])
    ]
    for label, entry in ranking + bounds:
        metric = entry["metric"]
        if not isinstance(metric, str) or metric not in metrics:
            errors.append(f"objective.{label}.metric {metric!r} is not declared in metrics")
        elif isinstance(metrics[metric], dict):
            extract = metrics[metric].get("extract")
            if isinstance(extract, dict) and extract.get("kind") in ADVISORY_EXTRACT_KINDS:
                errors.append(
                    f"objective.{label}.metric {metric!r} is extracted by a source heuristic; "
                    "an objective must read a measured or deterministic quantity"
                )
    for label, entry in ranking:
        if entry["direction"] not in ("minimize", "maximize"):
            errors.append(f"objective.{label}.direction must be minimize or maximize")
    for label, entry in bounds:
        if entry["operator"] not in OPERATORS:
            errors.append(f"objective.{label}.operator must be one of " + ", ".join(OPERATORS))
        if not isinstance(entry["value"], (int, float)) or isinstance(entry["value"], bool):
            errors.append(f"objective.{label}.value must be a number")
    for index, constraint in enumerate(objective["constraints"]):
        label = f"objective.constraints[{index}]"
        metric = constraint["metric"]
        if (constraint["operator"] in EQUALITY_OPERATORS and isinstance(metric, str)
                and metric in metrics and isinstance(metrics[metric], dict)
                and not metrics[metric].get("deterministic")):
            errors.append(
                f"{label} pins {metric!r} with '==' but the metric is not declared "
                "deterministic. Bound a variable quantity with '<=' instead"
            )
        if not isinstance(constraint["basis"], str) or not constraint["basis"].strip():
            errors.append(f"{label}.basis must say how the bound was set")
        if metric == objective["quality_gate"]["metric"]:
            errors.append(
                f"{label} names {metric!r}, the quality-gate metric: its threshold is "
                "objective.quality_gate, and two thresholds on one metric is ambiguous"
            )
        if metric == objective["target"]["metric"]:
            errors.append(
                f"{label} names {metric!r}, the optimization target: a constraint on the "
                "target truncates the search instead of bounding its side effects"
            )


def _reference(raw: Dict[str, Any], errors: List[str]) -> None:
    reference = raw["reference_launch"]
    metrics = raw["metrics"] if isinstance(raw.get("metrics"), dict) else {}
    for key in sorted(set(reference) - {"position", "assert_exact"}):
        errors.append(f"reference_launch.{key} is not a reference-launch field")
    if reference.get("position") != 1:
        errors.append(
            "reference_launch.position must be 1: the unmodified substrate is the run's "
            "required initial measurement of its own conditions"
        )
    asserts = reference.get("assert_exact")
    if not isinstance(asserts, dict) or not asserts:
        errors.append("reference_launch.assert_exact must name at least one quantity")
        asserts = {}
    for metric, value in asserts.items():
        if metric not in metrics:
            errors.append(f"reference_launch.assert_exact: {metric!r} is not a declared metric")
            continue
        if not metrics[metric].get("deterministic"):
            errors.append(
                f"reference_launch.assert_exact: {metric!r} is not declared deterministic, so "
                "it is a measured quantity; only a quantity fixed by the code may be asserted "
                "exactly, because anything else would need a tolerance -- and a tolerance needs "
                "a dispersion estimate this benchmark does not measure"
            )
        if not isinstance(value, (int, float, bool)):
            errors.append(f"reference_launch.assert_exact.{metric} must be a number or boolean")


def _report(raw: Dict[str, Any], errors: List[str]) -> None:
    report = raw["report"]
    metrics = raw["metrics"] if isinstance(raw.get("metrics"), dict) else {}
    headline = report.get("headline")
    if headline not in metrics:
        errors.append(f"report.headline {headline!r} is not a declared metric")
    columns = report.get("columns")
    if not isinstance(columns, list) or not columns:
        errors.append("report.columns must be a non-empty list")
        return
    for column in columns:
        if column not in metrics:
            errors.append(f"report.columns: {column!r} is not a declared metric")
    if headline in metrics and headline not in columns:
        errors.append("report.columns must include the headline metric")


def _placeholders(raw: Dict[str, Any], errors: List[str], prefix: str = "") -> None:
    if isinstance(raw, dict):
        for key, value in raw.items():
            _placeholders(value, errors, f"{prefix}.{key}" if prefix else key)
    elif isinstance(raw, list):
        for index, value in enumerate(raw):
            _placeholders(value, errors, f"{prefix}[{index}]")
    elif isinstance(raw, str):
        lowered = raw.lower()
        for marker in PLACEHOLDER_MARKERS:
            if marker.lower() in lowered:
                errors.append(
                    f"{prefix}: unresolved placeholder {raw!r}; a task fails closed rather "
                    "than running against an invented value"
                )
                break
