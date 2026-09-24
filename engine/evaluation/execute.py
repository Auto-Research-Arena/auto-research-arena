"""Execute one launch and append its measured outcome.

Check immutable files before starting work, use the reserved batch intent or
write a reference intent, then execute and extract this launch's own evidence.
Status and charging remain separate from task eligibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from engine.evaluation.task import Task
from engine.records.charging import Charge, charge as charge_rule
from engine.records.ledger import Ledger
from engine.evaluation.metrics import extract_metrics

from engine.compute.backend import LaunchOutcome, LaunchRequest, Worker

# Markers that distinguish an out-of-memory failure from any other crash. Worth
# separating: OOM is search information about where the memory wall is, and a method that
# cannot tell it from a syntax error will keep walking into it.
OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "torch.cuda.outofmemoryerror",
    "cublas_status_alloc_failed",
)


class ImmutableBoundaryError(Exception):
    """A candidate edited a file the task declares immutable.

    Execution is refused. Batch admission checks before reserving an intent;
    this later check can raise after a batch intent has already been written.
    """


def immutable_edits(task: Task, source_root: Path) -> list:
    """Compare existing immutable files with the task package.

    The runtime separately requires all declared files when capturing source
    identity. This byte comparison does not establish source completeness.
    """
    pinned = task.source_root
    source_root = Path(source_root)
    if pinned.resolve() == source_root.resolve():
        return []  # the pinned tree itself, launched directly: nothing was copied to diverge
    out = []
    for name in task.substrate.get("immutable") or []:
        original, candidate = pinned / name, source_root / name
        if not original.is_file():
            continue
        if not candidate.is_file():
            continue
        if candidate.read_bytes() != original.read_bytes():
            out.append(name)
    return sorted(out)


@dataclass
class LaunchRecord:
    """What one launch produced. Everything here is already in the ledger."""

    launch_seq: int
    launch_uuid: str
    candidate_id: str
    status: str
    witness: bool
    metrics: Dict[str, Optional[float]]
    extraction_errors: list
    charge: Charge
    outcome: LaunchOutcome
    stdout_path: Path

    @property
    def charged(self) -> bool:
        return self.charge.charged


def execute(
    *,
    task: Task,
    ledger: Ledger,
    worker: Worker,
    candidate_id: str,
    source_root: Path,
    log_dir: Path,
    parent_id: Optional[str] = None,
    idea: Optional[Mapping[str, Any]] = None,
    is_reference_launch: bool = False,
    launch_seq: Optional[int] = None,
    backend_name: str = "",
    prewritten_intent: Optional[Mapping[str, Any]] = None,
) -> LaunchRecord:
    """Record normal command failures as outcomes, including OOM and timeouts.

    Invalid input, compute substrate failures and unexpected extraction errors
    may propagate. An intent without a result remains conservatively charged.
    """
    source_root = Path(source_root).resolve()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"

    if prewritten_intent is not None:
        if prewritten_intent.get("candidate_id") != candidate_id:
            raise ValueError("prewritten intent belongs to a different candidate")
        seq = int(prewritten_intent["launch_seq"])
        if launch_seq is not None and seq != launch_seq:
            raise ValueError("prewritten intent launch_seq disagrees with launch_seq")
    else:
        seq = launch_seq if launch_seq is not None else ledger.next_launch_seq()

    # Recheck the instrument immediately before execution. A batch may have
    # reserved its intent since the earlier admission check.
    edited = immutable_edits(task, source_root)
    if edited:
        raise ImmutableBoundaryError(
            f"{candidate_id} edited {', '.join(edited)}, which "
            f"{task.task_id} declares immutable. Execution refused; restore the file "
            "from the task code directory."
        )

    # Record the intent before work starts, or use the batch's reserved intent.
    # An interruption then leaves a detectable unmatched launch.
    intent = dict(prewritten_intent) if prewritten_intent is not None else ledger.intent(
        candidate_id,
        launch_seq=seq,
        parent_id=parent_id,
        is_reference_launch=is_reference_launch,
        compute_backend=backend_name,
        compute_handle=worker.worker_id,
        compute_run_id=worker.compute_run_id,
        lane=worker.worker_id,
        idea=idea,
    )

    request = LaunchRequest(
        launch_id=intent["launch_uuid"],
        command=list(task.substrate["entrypoint"]),
        cwd=str(source_root),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        # Instrument settings reach reference and candidate launches alike.
        # The worker applies them over its configured environment defaults.
        env=dict(task.launch.get("env") or {}),
        gpus=int(task.launch.get("gpus_per_launch", 1)),
        timeout_seconds=task.launch.get("timeout_seconds"),
    )

    outcome = worker.run(request)

    # This launch's own stdout, read from the path the request named -- not from
    # a shared log, and not from the previous attempt's file. A witness read from
    # somebody else's output charges a launch that never touched a GPU.
    stdout = _read(Path(outcome.stdout_path))
    stderr = _read(Path(outcome.stderr_path))
    witness = task.gpu_work_witness in stdout

    # `source_root` is the candidate's copy, because a metric read from a
    # constant in a mutable file is reading part of what the candidate changed.
    metrics, errors = extract_metrics(task, stdout, source_root=source_root)

    status = classify(task, outcome, witness, metrics, stdout, stderr)

    result = ledger.result(
        intent["launch_uuid"],
        status=status,
        witness=witness,
        metrics=metrics,
        exit_code=outcome.exit_code,
        wall_time_seconds=outcome.wall_time_seconds,
        extraction_errors=errors,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        notes=outcome.notes,
    )

    return LaunchRecord(
        launch_seq=seq,
        launch_uuid=intent["launch_uuid"],
        candidate_id=candidate_id,
        status=status,
        witness=witness,
        metrics=metrics,
        extraction_errors=errors,
        charge=charge_rule(result),
        outcome=outcome,
        stdout_path=stdout_path,
    )


def classify(
    task: Task,
    outcome: LaunchOutcome,
    witness: bool,
    metrics: Mapping[str, Optional[float]],
    stdout: str,
    stderr: str,
) -> str:
    """Map an outcome onto one of the ledger's statuses.

    Precedence is fixed, and the first two clauses are ordered the way they are because
    `launched` and `exit_code is None` are different facts. A preempted launch has no exit
    code *and* did start; a refused one has no exit code and did not. Collapsing them
    charges a refusal or refunds a preemption, and both are budget errors that never
    surface as an error message.

    The status is a description, not a charging decision -- `engine.records.charging` reads the
    witness, and the two must stay separable so a status taxonomy can grow without
    silently changing what the budget counts.
    """
    if not outcome.launched:
        return "never_executed"
    if outcome.preempted:
        return "preempted"
    if outcome.timed_out:
        return "timeout"

    if not witness:
        # The command ran and never reached the GPU: an import error, a missing dataset,
        # a broken environment. Charging this would bill the method for our plumbing.
        return "substrate_failure"

    haystack = (stdout + "\n" + stderr).lower()
    if outcome.exit_code != 0:
        if any(marker in haystack for marker in OOM_MARKERS):
            return "oom"
        return "crash"

    # Exit zero with the witness present. Almost always fine, but a diverged run also
    # lands here: it trains, prints a summary and exits cleanly with a NaN in it.
    quality = task.objective_spec.get("quality_gate", task.objective_spec["target"])
    headline = metrics.get(quality["metric"])
    if headline is None or _not_finite(headline):
        return "fail_loss"
    return "ok"


def _not_finite(value: Any) -> bool:
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _read(path: Path) -> str:
    """Read a log, tolerating absence and undecodable bytes.

    `errors="replace"` rather than strict: a CUDA driver message with a stray byte in it
    would otherwise raise here, *after* the launch, and turn a recorded crash into an
    unrecorded one. The witness test is a substring search and survives replacement.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, FileNotFoundError):
        return ""
