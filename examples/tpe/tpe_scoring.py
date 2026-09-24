"""One reference-normalized TPE score for every task."""

import math
import operator


COMPARISONS = {"<": operator.lt, "<=": operator.le, "==": operator.eq,
               ">": operator.gt, ">=": operator.ge}

try:
    from autoarena import objective_spec
except ModuleNotFoundError:
    from engine.autoarena import objective_spec


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def scalar(result, task, reference):
    """Minimize the target/reference ratio with shared constraint penalties.

    Canonical measurements and benchmark eligibility remain engine-owned.
    """
    objective = objective_spec(task["objective"])
    metric = objective["target"]["metric"]
    gate = objective["quality_gate"]
    metrics = result.get("metrics") or {}
    quality, target = metrics.get(gate["metric"]), metrics.get(metric)
    if result.get("status") != "ok" or not _finite(quality) or not _finite(target):
        return 3000.0, "no_measurement"
    for clause in objective["constraints"]:
        observed = metrics.get(clause["metric"])
        if not _finite(observed) or not COMPARISONS[clause["operator"]](observed, clause["value"]):
            return 2000.0, "inadmissible"
    if target <= 0:
        return 2000.0, "inadmissible"
    if not COMPARISONS[gate["operator"]](quality, gate["value"]):
        return 1000.0 + abs(quality - gate["value"]), "gate_breach"
    scale = (reference.get("metrics") or {}).get(metric)
    if not _finite(scale) or scale <= 0:
        return 3000.0, "no_reference"
    return target / scale, "admissible"
