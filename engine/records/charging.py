"""Charge launches with the task's GPU-work marker; otherwise do not charge.

Unresolved launches count until a result is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

CHARGED = "CHARGED"
REFUNDED = "NOT_CHARGED"


@dataclass(frozen=True)
class Charge:
    decision: str
    reason: str
    classification: str

    @property
    def charged(self) -> bool:
        return self.decision == CHARGED


def charge(result_row: Optional[Mapping[str, Any]]) -> Charge:
    """Classify a launch using its latest result row, or None if unresolved."""
    if result_row is None:
        return Charge(
            CHARGED,
            "No final result; budget slot reserved.",
            "unresolved",
        )

    # Check the marker before exit_code: interrupted GPU work still counts.
    if bool(result_row.get("witness")):
        status = str(result_row.get("status") or "unknown")
        return Charge(
            CHARGED,
            f"GPU-work marker present (status: {status}).",
            "gpu_work_no_measurement" if result_row.get("status") != "ok" else "measured",
        )

    if result_row.get("exit_code") is None:
        return Charge(
            REFUNDED,
            "No GPU-work marker or exit code recorded.",
            "never_executed",
        )

    return Charge(
        REFUNDED,
        f"No GPU-work marker; exit code {result_row.get('exit_code')}.",
        "substrate_failure",
    )
