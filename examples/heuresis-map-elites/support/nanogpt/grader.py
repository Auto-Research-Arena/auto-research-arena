"""Grade canonical stdout using the task's target and eligibility rules.

The socket tool and host harness both call grade(). The displayed score is the
raw target; archive fitness is computed separately by objective.evaluate().
"""

from __future__ import annotations

from typing import Any

from heuresis.grading import GradingServer


def _parse_run_log(text: str) -> dict[str, Any]:
    from heuresis.tasks.nanogpt import objective as obj

    metrics = obj.parse_metrics_json(text)
    if metrics is None:
        return {
            "score": None,
            "valid": False,
            "details": {"error": "No METRICS_JSON line in run.log. The run did not "
                                 "reach the end; read the traceback in run.log."},
        }

    the_task = obj.task()
    _, arena = obj._autoarena()
    rank_metric = obj.ranking_metric()
    gate_metric = obj.gate_metric()

    verdict = arena.admissibility(the_task, metrics)
    details: dict[str, Any] = {
        "is_lower_better": True,
        "ranking_key": rank_metric,
        gate_metric: metrics.get(gate_metric),
        "admissible": bool(verdict.admissible),
    }
    if not verdict.admissible:
        details["reject_clause"] = verdict.clause
        details["reject_reason"] = verdict.reason
    else:
        details["gate_margin"] = arena.gate_margin(the_task, metrics)

    # Carried through for the record; every one comes from the one metric line.
    for name in ("flops_per_token_measured", "training_data_tokens_available",
                 "num_params_total", "num_params_active", "total_tokens",
                 "num_steps", "training_seconds", "peak_vram_reserved_bytes",
                 "train_tokens_per_second"):
        if name in metrics:
            details[name] = metrics[name]

    score = metrics.get(rank_metric)
    if score is None:
        return {"score": None, "valid": False,
                "details": details | {
                    "error": f"METRICS_JSON carried no {rank_metric}"}}

    return {
        "score": float(score),
        # Admissibility, not "a number was printed". An inadmissible run is a real
        # measurement and a real rejection.
        "valid": bool(verdict.admissible),
        "details": details,
    }


class NanoGPTGrader(GradingServer):
    """Grades a run.log from nanoGPT training on the task's declared ranking key."""

    input_files = ["run.log"]

    def grade(self, files: dict[str, bytes]) -> dict[str, Any]:
        if "run.log" not in files:
            return {
                "score": None,
                "valid": False,
                "details": {"error": "No run.log found. Dispatch a launch first; the "
                                     "dispatch script writes run.log itself."},
            }
        try:
            text = files["run.log"].decode(errors="replace")
        except Exception as error:  # noqa: BLE001
            return {"score": None, "valid": False,
                    "details": {"error": f"Could not decode run.log: {error}"}}
        return _parse_run_log(text)
