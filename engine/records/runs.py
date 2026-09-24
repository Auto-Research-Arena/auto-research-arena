"""Run directories and evidence locations.

`run.json` records identity and frozen task/method interfaces; `launches.jsonl`
records measurements. Verification also reads launch logs and captured sources
under `engine/source-snapshots/`. Runtime owns dispatch state and the captured
method interface; a method's workspace is separate from measurement evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.records.ledger import Ledger
from engine.evaluation.task import Task, load_task
from engine.records.ledger import launch_view
from engine import ROOT

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
RUN_JSON = "run.json"
LEDGER_NAME = "launches.jsonl"


class RunError(Exception):
    """A run directory is unusable, or would be corrupted by this operation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Run:
    """One run directory, opened. Cheap to construct; reads on demand."""

    path: Path
    meta: Dict[str, Any]

    # -- identity ----------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.meta["run_id"]

    @property
    def task_id(self) -> str:
        return self.meta["task_id"]

    @property
    def method_id(self) -> str:
        return self.meta["method_id"]

    @property
    def ledger_path(self) -> Path:
        return self.path / LEDGER_NAME

    def ledger(self) -> Ledger:
        return Ledger(self.ledger_path)

    def views(self) -> List[Any]:
        return launch_view(self.ledger().rows())

    def task(self) -> Task:
        """Load current rules for comparison with the run's recorded task."""
        return load_task(self.task_id)

    def recorded_task(self) -> Dict[str, Any]:
        """The task definition as it stood when the run started."""
        return self.meta["task_snapshot"]

    def log_dir(self, launch_seq: int, attempt: int = 1) -> Path:
        return self.path / "logs" / f"{launch_seq:04d}" / f"attempt-{attempt}"


# ---------------------------------------------------------------------------
# creating and opening
# ---------------------------------------------------------------------------


def create(
    root: Path,
    *,
    task: Task,
    method_id: str,
    run_id: str,
    compute: Optional[Dict[str, Any]] = None,
    research_model: Optional[str] = None,
    lanes: int = 1,
    notes: str = "",
    method_snapshot: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Run:
    """Create metadata with frozen interfaces; refuse to reuse an existing ledger."""
    if not RUN_ID_PATTERN.match(run_id):
        raise RunError(
            f"run_id {run_id!r} must be alphanumeric with . _ - and no path separators; "
            "it names a directory and appears in every artifact"
        )
    path = Path(root) / run_id
    if (path / LEDGER_NAME).exists():
        raise RunError(
            f"{path / LEDGER_NAME} already exists. Pick a new run_id: a ledger shared by "
            "two runs derives a launch count that is the sum of two budgets, and nothing "
            "in the file records that it happened"
        )
    path.mkdir(parents=True, exist_ok=True)
    reserved = {
        "schema", "run_id", "task_id", "task_version", "method_id", "lanes",
        "created_at", "created_by", "hostname", "harness_revision", "compute",
        "research_model",
        "notes", "task_snapshot", "task_snapshot_sha256", "method_snapshot",
        "method_snapshot_sha256",
    }
    overlap = sorted(set(extra or {}) & reserved)
    if overlap:
        raise RunError(
            "run metadata extra may not replace authoritative field(s): "
            + ", ".join(overlap)
        )
    task_snapshot = copy.deepcopy(task.raw)
    meta = {
        "schema": "autoarena/run/1",
        "run_id": run_id,
        "task_id": task.task_id,
        "task_version": task.version,
        "method_id": method_id,
        "lanes": int(lanes),
        "created_at": utc_now(),
        "created_by": os.environ.get("USER") or "unknown",
        "hostname": os.uname().nodename,
        "harness_revision": git_revision(ROOT),
        "compute": dict(compute or {}),
        "research_model": research_model,
        "notes": notes,
        # Copy the definition so later in-process edits cannot mutate the snapshot.
        "task_snapshot": task_snapshot,
        "task_snapshot_sha256": _json_sha256(task_snapshot),
    }
    if method_snapshot is not None:
        frozen_method = copy.deepcopy(method_snapshot)
        meta["method_snapshot"] = frozen_method
        meta["method_snapshot_sha256"] = _json_sha256(frozen_method)
    meta.update(extra or {})
    (path / RUN_JSON).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return Run(path=path, meta=meta)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def open_run(path: Path) -> Run:
    path = Path(path)
    meta_path = path / RUN_JSON
    if not meta_path.is_file():
        raise RunError(
            f"{path} is not a run directory: no {RUN_JSON}. Every downstream command reads "
            "the task and method from it, and inferring them from a directory name is how "
            "a run gets reported under the wrong task"
        )
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as error:
        raise RunError(f"{meta_path}: not valid JSON: {error}") from error
    for field in ("run_id", "task_id", "method_id"):
        if not meta.get(field):
            raise RunError(f"{meta_path}: missing {field!r}")
    return Run(path=path, meta=meta)


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------


def git_revision(directory: Path) -> Optional[str]:
    """Return Git HEAD, or None on an unsuccessful probe; process errors propagate."""
    probe = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.stdout.strip() or None if probe.returncode == 0 else None


def log_path(run, recorded):
    """Resolve a run-owned log after moving or copying its run directory."""
    text = str(recorded)
    if "/logs/" in text:
        return run.path / text[text.index("/logs/") + 1:]
    path = Path(text)
    return path if path.is_absolute() else run.path / path


def candidate_source(run, candidate_id):
    """Return the candidate source captured at measurement time."""
    return run.path / "engine/source-snapshots" / candidate_id
