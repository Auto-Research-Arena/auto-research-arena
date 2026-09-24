"""Admissibility and comparison: the arithmetic that decides a candidate.

This is the only place a decision is made, and it is deliberately mechanical --
two recorded numbers and an operator. No plausibility, no effect size, no
mechanism, no tolerance. A method may reason however it likes about what to try
next; it does not get to reason about whether its candidate won.

The decision order is:

    1. ceilings and the gate      -> inadmissible, and the target is never compared
    2. target and tiebreaks       -> accept or reject
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

Metrics = Mapping[str, Any]


@dataclass(frozen=True)
class Verdict:
    """The outcome of one comparison, with the reason a reader can check."""

    accepted: bool
    admissible: bool
    clause: str
    reason: str
    compared: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:  # pragma: no cover - convenience only
        return self.accepted


# ---------------------------------------------------------------------------
# admissibility: ceilings and the gate, before anything is compared
# ---------------------------------------------------------------------------


def admissibility(task: Any, metrics: Metrics) -> Verdict:
    """Ceilings first, then the gate. A breach names the clause that fired.

    Naming the clause is not cosmetic: "rejected: inadmissible" cannot later be
    distinguished from a target regression, and a run pinned against one ceiling
    for twenty launches looks exactly like a run that ran out of ideas.
    """
    objective = task.objective_spec
    for index, ceiling in enumerate(objective["constraints"]):
        metric = ceiling["metric"]
        value = metrics.get(metric)
        if value is None:
            return Verdict(
                False,
                False,
                f"ceiling[{index}]:{metric}:unmeasured",
                f"{metric} was not measured, so the ceiling {metric} "
                f"{ceiling['operator']} {ceiling['value']} cannot be evaluated; an "
                "unevaluable ceiling is a refusal, not a pass",
                {metric: None, "ceiling": ceiling["value"]},
            )
        if not satisfies(float(value), ceiling["operator"], float(ceiling["value"])):
            return Verdict(
                False,
                False,
                f"ceiling[{index}]:{metric}",
                f"inadmissible: {metric} = {value} breaches the ceiling "
                f"{ceiling['operator']} {ceiling['value']} ({ceiling['basis']})",
                {metric: value, "ceiling": ceiling["value"]},
            )

    gate = objective["quality_gate"]
    value = metrics.get(gate["metric"])
    if value is None:
        return Verdict(
            False,
            False,
            f"gate:{gate['metric']}:unmeasured",
            f"{gate['metric']} was not measured, so the gate cannot be evaluated",
            {gate["metric"]: None, "gate": gate["value"]},
        )
    if not satisfies(float(value), gate["operator"], float(gate["value"])):
        return Verdict(
            False,
            False,
            f"gate:{gate['metric']}",
            f"inadmissible: {gate['metric']} = {value} fails the gate "
            f"{gate['operator']} {gate['value']}; the target is not compared at all",
            {gate["metric"]: value, "gate": gate["value"]},
        )

    floor = _validity_floor(task, metrics)
    if floor is not None:
        return floor

    return Verdict(True, True, "admissible", "every ceiling and the gate are satisfied", {})


def _validity_floor(task: Any, metrics: Metrics) -> Optional[Verdict]:
    """A ranking target must be strictly positive and finite. Universal, every task.

    This is not fussiness about numbers, it is the clause that stops an axis from being
    won by deleting the thing it measures. Every one of these has happened: an
    arithmetic-cost axis reached exactly 0 by routing the work through operations the
    dispatch counter does not register, and three frontier slots then tied at 0 with
    nothing left to improve; a decode-state axis reached 0 by holding no state at all.
    Zero is also terminal under minimise-with-strict-improvement, so admitting it ends
    the search rather than winning it.

    A `None` target is not handled here. That is unmeasured, which the comparison
    refuses with its own reason, and on an axis whose target only exists once the
    candidate builds the mechanism -- a decode cache -- unmeasured is the honest
    starting state of the unmodified substrate rather than a breach.
    """
    for entry in task.ranking_metrics:
        metric = entry["metric"]
        value = metrics.get(metric)
        if value is None:
            continue
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return Verdict(
                False,
                False,
                f"validity_floor:{metric}:non_finite",
                f"inadmissible: {metric} = {value} is not finite, so it cannot order anything",
                {metric: value},
            )
        if number <= 0:
            return Verdict(
                False,
                False,
                f"validity_floor:{metric}:non_positive",
                f"inadmissible: {metric} = {value} is not strictly positive. A target at "
                "zero means the quantity being measured is absent rather than small, and "
                "it is terminal under strict improvement -- nothing can beat it",
                {metric: value},
            )
    return None


def gate_margin(task: Any, metrics: Metrics) -> Optional[float]:
    """Signed distance from the gate threshold, positive when inside.

    Report the margin, not just pass/fail. A candidate flush against the threshold
    has no currency left to spend, which means the run has quietly become `strict`
    on the target alone -- and that is invisible from a boolean.
    """
    gate = task.objective_spec["quality_gate"]
    value = metrics.get(gate["metric"])
    if value is None:
        return None
    bound = float(gate["value"])
    if gate["operator"] in ("<", "<="):
        return bound - float(value)
    return float(value) - bound


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def compare(task: Any, candidate: Metrics, incumbent: Metrics) -> Verdict:
    """Check eligibility, then compare the target and its ordered tiebreaks."""
    verdict = admissibility(task, candidate)
    if not verdict.admissible:
        return verdict

    entries = task.ranking_metrics
    target = entries[0]
    metric, direction = target["metric"], target["direction"]
    mine, theirs = _pair(metric, candidate, incumbent)
    if mine is None or theirs is None:
        return Verdict(
            False,
            True,
            f"unmeasurable:{metric}",
            f"{metric} is the ranking key and is missing on one side ({mine} vs {theirs})",
            {metric: [mine, theirs]},
        )
    compared = {metric: [mine, theirs]}
    if _better(mine, theirs, direction):
        return Verdict(
            True,
            True,
            f"gated:target:{metric}",
            f"accept: {metric} {mine} strictly improves on {theirs} ({direction}); the "
            "quality metric is inside its gate and is not compared",
            compared,
        )
    if mine != theirs:
        return Verdict(
            False,
            True,
            f"gated:regression:{metric}",
            f"reject: {metric} {mine} is worse than {theirs} ({direction})",
            compared,
        )

    # Exact tie on the target. Later entries break it in listed order.
    for index, entry in enumerate(entries[1:], start=1):
        tie_metric, tie_direction = entry["metric"], entry["direction"]
        tie_mine, tie_theirs = _pair(tie_metric, candidate, incumbent)
        if tie_mine is None or tie_theirs is None:
            continue
        compared[tie_metric] = [tie_mine, tie_theirs]
        if _better(tie_mine, tie_theirs, tie_direction):
            return Verdict(
                True,
                True,
                f"gated:tiebreak[{index}]:{tie_metric}",
                f"accept: {metric} is exactly equal ({mine}) and {tie_metric} {tie_mine} "
                f"strictly improves on {tie_theirs} ({tie_direction})",
                compared,
            )
        if tie_mine != tie_theirs:
            return Verdict(
                False,
                True,
                f"gated:tiebreak[{index}]:regression:{tie_metric}",
                f"reject: {metric} is exactly equal ({mine}) and {tie_metric} {tie_mine} is "
                f"worse than {tie_theirs} ({tie_direction})",
                compared,
            )
    return Verdict(
        False,
        True,
        "gated:tie",
        f"reject: {metric} is exactly equal ({mine}) and no tie-break improves. Equality is "
        "not an accept; a method that wants an objective-neutral candidate to count is "
        "applying its own policy on top of this, and must record it as such",
        compared,
    )


def rank_key(task: Any, metrics: Metrics) -> Optional[tuple]:
    """A sort key over admissible candidates, best first.

    `None` means the candidate is inadmissible or unmeasurable, and it must be kept
    out of the map a ranking is computed over -- not ranked last. On a minimized
    axis a breacher sits exactly where the most attractive values are, so an
    inadmissible entry left among the values wins a naive `min()`.
    """
    if not admissibility(task, metrics).admissible:
        return None
    key: List[float] = []
    for entry in task.ranking_metrics:
        value = metrics.get(entry["metric"])
        if value is None:
            return None
        signed = float(value) if entry["direction"] == "minimize" else -float(value)
        key.append(signed)
    margin = gate_margin(task, metrics)
    if margin is not None:
        # Larger margin first on an exact tie of every ranking metric: it prefers
        # the candidate with more currency left to spend in later rounds.
        key.append(-margin)
    return tuple(key)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def satisfies(value: float, operator: str, bound: float) -> bool:
    if operator == "<":
        return value < bound
    if operator == "<=":
        return value <= bound
    if operator == ">":
        return value > bound
    if operator == ">=":
        return value >= bound
    if operator == "==":
        # Exact, deliberately. `==` states that a quantity is part of the immutable
        # boundary rather than something to be bounded, so any deviation at all is a
        # different task. Task validation refuses `==` on a metric that is not
        # declared deterministic, because a metric with run-to-run variation could
        # never satisfy it and every candidate would be rejected for a true reason
        # that says nothing about the candidate.
        return value == bound
    raise ValueError(
        f"unrecognised operator {operator!r}: refuse rather than guess. At the boundary "
        "'<' and '<=' disagree, and a ranking key that rewards approaching the threshold "
        "makes a candidate exactly at it a reachable case"
    )


def _better(left: Any, right: Any, direction: str) -> bool:
    if direction == "minimize":
        return float(left) < float(right)
    if direction == "maximize":
        return float(left) > float(right)
    raise ValueError(f"unrecognised direction {direction!r}")


def _pair(metric: str, candidate: Metrics, incumbent: Metrics):
    return _numeric(metric, candidate.get(metric)), _numeric(metric, incumbent.get(metric))


def _numeric(metric: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{metric} arrived as {value!r}, which is not a number. Coerce explicitly and "
            "fail loudly: a numeric field arriving as a string has produced NaN comparisons "
            "that silently rejected every result with a plausible-looking reason"
        ) from None
