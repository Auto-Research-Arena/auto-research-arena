"""Append-only launch evidence: intent, result, and adjudication rows.

The engine writes intents before execution and results afterward. Corrections
append adjudications; folding derives effective outcomes and charge decisions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from engine.records.charging import Charge, charge

ROW_TYPES = ("intent", "result", "adjudication")
STATUSES = (
    "ok",
    "oom",
    "crash",
    "fail_loss",
    "timeout",
    "preempted",
    "never_executed",
    "substrate_failure",
)


class LedgerError(Exception):
    """The ledger is malformed, or an append would corrupt it."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only reader/writer for one run's `launches.jsonl`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- reading -----------------------------------------------------------
    def rows(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        for number, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise LedgerError(f"{self.path}:{number}: not valid JSON: {error}") from error
            if not isinstance(row, dict) or row.get("record_type") not in ROW_TYPES:
                raise LedgerError(
                    f"{self.path}:{number}: record_type must be one of {', '.join(ROW_TYPES)}"
                )
            rows.append(row)
        return rows

    def next_launch_seq(self) -> int:
        """The next attempt ordinal. Never reused, so a refunded attempt keeps an
        identity that later rows can refer to."""
        seqs = [
            int(row["launch_seq"])
            for row in self.rows()
            if row["record_type"] == "intent" and "launch_seq" in row
        ]
        return max(seqs) + 1 if seqs else 1

    # -- writing -----------------------------------------------------------
    def append(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        validate_row(row)
        record = dict(row)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        return record

    def intent(
        self,
        candidate_id: str,
        *,
        launch_seq: Optional[int] = None,
        launch_uuid: Optional[str] = None,
        parent_id: Optional[str] = None,
        is_reference_launch: bool = False,
        compute_backend: Optional[str] = None,
        compute_handle: Optional[str] = None,
        compute_run_id: Optional[str] = None,
        lane: Optional[str] = None,
        idea: Optional[Mapping[str, Any]] = None,
        research_log: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write the intent row. Call this **before** the launch starts.

        Intent-first is what makes a preemption between the two writes leave a
        detectable, charged, unmatched launch instead of an invisible one.
        """
        row: Dict[str, Any] = {
            "record_type": "intent",
            "launch_seq": launch_seq if launch_seq is not None else self.next_launch_seq(),
            "launch_uuid": launch_uuid or str(uuid.uuid4()),
            "candidate_id": candidate_id,
            "parent_id": parent_id,
            "is_reference_launch": is_reference_launch,
            "started_at": now(),
            "compute_backend": compute_backend,
            "compute_handle": compute_handle,
            "compute_run_id": compute_run_id,
            "lane": lane,
        }
        if idea is not None:
            row["idea"] = dict(idea)
        if research_log is not None:
            row["research_log"] = dict(research_log)
        return self.append(row)

    def result(
        self,
        launch_uuid: str,
        *,
        status: str,
        witness: bool,
        metrics: Mapping[str, Any],
        exit_code: Optional[int] = None,
        wall_time_seconds: Optional[float] = None,
        extraction_errors: Optional[Iterable[str]] = None,
        stdout_path: Optional[str] = None,
        stderr_path: Optional[str] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        return self.append(
            {
                "record_type": "result",
                "launch_uuid": launch_uuid,
                "status": status,
                "witness": bool(witness),
                "exit_code": exit_code,
                "finished_at": now(),
                "wall_time_seconds": wall_time_seconds,
                "metrics": dict(metrics),
                "extraction_errors": list(extraction_errors or []),
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "notes": notes,
            }
        )

    def adjudicate(
        self,
        launch_uuid: str,
        *,
        decision: str,
        ground: Iterable[str],
        decided_by: str,
        supersedes: Optional[str] = None,
        abandoned: bool = False,
    ) -> Dict[str, Any]:
        """Append a charge correction with reasons.

        `abandoned=True` settles a launch whose result cannot arrive, without
        inventing a measurement. The decision still determines its charge.
        """
        # Reject strings before list() could turn one reason into characters.
        if isinstance(ground, str):
            raise LedgerError(
                "adjudicate(ground=...) requires a sequence of strings; "
                "pass [ground] for a single reason"
            )
        if not isinstance(abandoned, bool):
            raise LedgerError("adjudication.abandoned must be a boolean")
        return self.append(
            {
                "record_type": "adjudication",
                "launch_uuid": launch_uuid,
                "decision": decision,
                "ground": list(ground),
                "supersedes": supersedes,
                "decided_by": decided_by,
                "decided_at": now(),
                **({"abandoned": True} if abandoned else {}),
            }
        )


def validate_row(row: Mapping[str, Any]) -> None:
    kind = row.get("record_type")
    if kind not in ROW_TYPES:
        raise LedgerError(f"record_type must be one of {', '.join(ROW_TYPES)}")
    if kind == "intent":
        for field_name in ("launch_seq", "launch_uuid", "candidate_id", "started_at"):
            if not row.get(field_name):
                raise LedgerError(f"intent row requires {field_name!r}")
        if not isinstance(row["launch_seq"], int) or row["launch_seq"] < 1:
            raise LedgerError("intent.launch_seq must be a positive integer")
    elif kind == "result":
        if not row.get("launch_uuid"):
            raise LedgerError("result row requires 'launch_uuid'")
        if row.get("status") not in STATUSES:
            raise LedgerError(f"result.status must be one of {', '.join(STATUSES)}")
        if not isinstance(row.get("witness"), bool):
            raise LedgerError(
                "result.witness must be an explicit boolean: it decides charging, and an "
                "absent witness is not the same as a false one"
            )
        if not isinstance(row.get("metrics"), dict):
            raise LedgerError("result.metrics must be an object, with null for absent values")
    else:
        if not isinstance(row.get("launch_uuid"), str) or not row["launch_uuid"]:
            raise LedgerError("adjudication row requires a nonempty 'launch_uuid' string")
        if row.get("decision") not in ("CHARGED", "NOT_CHARGED"):
            raise LedgerError("adjudication.decision must be CHARGED or NOT_CHARGED")
        if not isinstance(row.get("ground"), list) or not row["ground"] or not all(
            isinstance(item, str) for item in row["ground"]
        ):
            raise LedgerError("adjudication.ground must be a nonempty list of strings")
        if "abandoned" in row and not isinstance(row["abandoned"], bool):
            raise LedgerError("adjudication.abandoned must be a boolean")


# ---------------------------------------------------------------------------
# reading: one launch, folded
# ---------------------------------------------------------------------------


@dataclass
class LaunchView:
    """Every row for one `launch_uuid`, folded into the current truth."""

    launch_uuid: str
    launch_seq: int
    candidate_id: str
    parent_id: Optional[str]
    is_reference_launch: bool
    intent: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    adjudications: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def metrics(self) -> Dict[str, Any]:
        return dict((self.result or {}).get("metrics") or {})

    @property
    def status(self) -> str:
        if self.result is None:
            return "unresolved"
        return str(self.result.get("status"))

    @property
    def abandoned(self) -> bool:
        """Explicit abandonment settles a missing outcome; a charge correction does not."""
        return any(row.get("abandoned") is True for row in self.adjudications)

    @property
    def resolved(self) -> bool:
        """A result arrived, or an abandonment recorded that none can arrive."""
        return self.result is not None or self.abandoned

    def charge(self) -> Charge:
        """The last validated adjudication overrides the current result's charge."""
        base = charge(self.result)
        if self.adjudications:
            last = self.adjudications[-1]
            return Charge(
                last["decision"],
                "adjudicated: " + "; ".join(last["ground"]),
                base.classification,
            )
        return base


def launch_view(rows: Iterable[Mapping[str, Any]]) -> List[LaunchView]:
    """Fold ledger rows into one view per launch, in `launch_seq` order.

    Later rows win: the last result row for a uuid is the current one, which is what
    the charging rule must be applied to.
    """
    views: Dict[str, LaunchView] = {}
    orphans: List[str] = []
    for row in rows:
        if row.get("record_type") == "adjudication":
            validate_row(row)
        uuid_value = row.get("launch_uuid")
        if not uuid_value:
            continue
        if row["record_type"] == "intent":
            views[uuid_value] = LaunchView(
                launch_uuid=uuid_value,
                launch_seq=int(row["launch_seq"]),
                candidate_id=row.get("candidate_id", ""),
                parent_id=row.get("parent_id"),
                is_reference_launch=bool(row.get("is_reference_launch")),
                intent=dict(row),
                result=None,
            )
        elif uuid_value not in views:
            orphans.append(uuid_value)
        elif row["record_type"] == "result":
            views[uuid_value].result = dict(row)
        else:
            views[uuid_value].adjudications.append(dict(row))
    if orphans:
        raise LedgerError(
            "rows reference launch_uuids with no intent row: "
            + ", ".join(sorted(set(orphans)))
            + ". Intent is written first precisely so this cannot happen; a result without "
            "one means a writer skipped phase 1 or two writers shared the file"
        )
    return sorted(views.values(), key=lambda view: view.launch_seq)
