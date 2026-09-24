"""Admit candidate batches, lease workers and return raw measurement facts."""

from __future__ import annotations

import json
import fcntl
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping

from engine.records.ledger import Ledger

from engine.compute import load_backend

from engine.evaluation.execute import execute, immutable_edits
from engine.evaluation.task import accelerator_error


EVALUATION_SCHEMA = "autoarena/evaluation/1"
CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")


class ProtocolError(Exception):
    """A candidate launch would violate the common method protocol."""


def research_payload(value):
    """Freeze a method's account without judging ideas or prescribing a workflow."""
    if not isinstance(value, Mapping):
        raise ProtocolError("research_log must be a JSON object with ideas")
    ideas, status = value.get("ideas"), value.get("status")
    if (not isinstance(ideas, list) or not ideas
            or any(not ((isinstance(idea, str) and idea.strip())
                       or (isinstance(idea, Mapping) and idea)) for idea in ideas)):
        raise ProtocolError("research_log.ideas must be a nonempty list of nonempty strings or objects")
    if "status" in value and (not isinstance(status, str) or not status.strip()):
        raise ProtocolError("research_log.status must be a nonempty string")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ProtocolError("research_log must contain finite JSON-serializable data") from error


def evaluate_batch(
    run: Any,
    *,
    candidates: list[Mapping[str, Any]],
    backend_config: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    """Measure a driver-selected batch on this run's recorded worker allocation.

    Every item supplies ``candidate_id``, ``source`` and ``research_log``.
    Parent lineage and idea are optional method metadata. A parent need not name
    a candidate measured by this run.
    The whole batch is admitted before any intent or compute acquisition. The shared
    run lock reserves its identities and budget against other dispatches; individual
    intents are written only when a worker is leased, with that worker's actual lane.
    Queued candidates therefore do not become unmatched launches on an interruption.
    Allocated GPU models must satisfy the task before any intent is written.
    Results retain input order and the single-evaluation wire format.
    """
    if not isinstance(candidates, list) or not candidates:
        raise ProtocolError("candidates must be a non-empty list")
    config = dict(backend_config)
    backend_name = str(config.get("backend") or "")
    if not backend_name:
        raise ProtocolError("backend_config must name a backend")
    lanes = run.meta.get("lanes", 1)
    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 1:
        raise ProtocolError("the run must record a positive integer lane count")

    with _exclusive(run.path / ".lane-compute.lock"):
        task = _checked_task(run)
        work = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ProtocolError("each batch candidate must be an object")
            candidate_id = candidate.get("candidate_id")
            source = candidate.get("source")
            if not isinstance(source, (str, Path)) or not str(source):
                raise ProtocolError("each batch candidate must supply its source directory")
            source_root = _checked_source(task, candidate_id, Path(source))
            parent_id = candidate.get("parent_id")
            if parent_id is not None and (not isinstance(parent_id, str) or not parent_id.strip()):
                raise ProtocolError("parent_id must be a nonblank string when supplied")
            idea = candidate.get("idea", {})
            if not isinstance(idea, Mapping):
                raise ProtocolError("idea must be a JSON object when supplied")
            try:
                # Freeze caller-owned dictionaries and reject unserializable input before
                # any other member of a malformed batch can acquire a ledger identity.
                idea = json.loads(json.dumps(dict(idea), allow_nan=False))
            except (TypeError, ValueError) as error:
                raise ProtocolError("candidate idea must be a JSON object") from error
            work.append({
                "candidate_id": candidate_id,
                "source_root": source_root,
                "parent_id": parent_id,
                "idea": idea,
                "research_log": research_payload(candidate.get("research_log")),
            })

        with _exclusive(run.path / ".dispatch.lock"):
            views = run.views()
            _check_admission(task, views, [item["candidate_id"] for item in work])
            base = run.ledger().next_launch_seq()
            for offset, item in enumerate(work):
                item["launch_seq"] = base + offset

        ledger = _SerializedLedger(run.ledger_path, run.path / ".dispatch.lock")
        backend = load_backend(config)
        try:
            workers = backend.workers(
                lanes=lanes, gpus_per_lane=int(task.launch["gpus_per_launch"])
            )
            if len(workers) != lanes:
                raise ProtocolError("backend did not supply the run's recorded lane count")
            worker_ids = [worker.worker_id for worker in workers]
            if len(set(worker_ids)) != len(worker_ids):
                raise ProtocolError("backend supplied duplicate worker identities")
            gpu_ids = [gpu_id for worker in workers for gpu_id in worker.gpu_ids]
            problem = accelerator_error(task, gpu_ids, backend.gpu_names(gpu_ids))
            if problem:
                raise ProtocolError(problem)
            available: queue.Queue = queue.Queue()
            for worker in workers:
                available.put(worker)

            def one(item: Mapping[str, Any]) -> Dict[str, Any]:
                worker = available.get()
                try:
                    intent = ledger.intent(
                        item["candidate_id"],
                        launch_seq=item["launch_seq"],
                        parent_id=item["parent_id"],
                        compute_backend=backend_name,
                        compute_handle=worker.worker_id,
                        compute_run_id=worker.compute_run_id,
                        lane=worker.worker_id,
                        idea=item["idea"],
                        research_log=item["research_log"],
                    )
                    record = execute(
                        task=task,
                        ledger=ledger,
                        worker=worker,
                        candidate_id=item["candidate_id"],
                        source_root=item["source_root"],
                        log_dir=run.log_dir(item["launch_seq"]),
                        launch_seq=item["launch_seq"],
                        prewritten_intent=intent,
                    )
                    return _response(run, record)
                finally:
                    available.put(worker)

            with ThreadPoolExecutor(max_workers=lanes) as pool:
                return list(pool.map(one, work))
        finally:
            backend.release()


class _SerializedLedger(Ledger):
    """One writer at a time, including large result rows from concurrent workers."""

    def __init__(self, path: Path, lock_path: Path) -> None:
        super().__init__(path)
        self.lock_path = lock_path
        self.guard = threading.Lock()

    def append(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        with self.guard, _exclusive(self.lock_path):
            return super().append(row)


def _checked_task(run: Any) -> Any:
    task = run.task()
    if _canonical(run.recorded_task()) != _canonical(task.raw):
        raise ProtocolError(
            f"{task.task_id} changed after {run.run_id} started. Refusing a launch under a "
            "different task contract"
        )
    return task


def _checked_source(task: Any, candidate_id: Any, source_root: Path) -> Path:
    if not isinstance(candidate_id, str) or not CANDIDATE_ID.fullmatch(candidate_id):
        raise ProtocolError(
            "candidate_id must be alphanumeric with only '.', '_' or '-', and at most 121 chars"
        )
    source_root = Path(source_root).resolve()
    if not source_root.is_dir():
        raise ProtocolError(f"candidate source directory does not exist: {source_root}")

    edited = immutable_edits(task, source_root)
    if edited:
        raise ProtocolError(
            f"candidate {candidate_id!r} edits immutable file(s): {', '.join(edited)}"
        )

    return source_root


def _response(run: Any, record: Any) -> Dict[str, Any]:
    return {
        "schema": EVALUATION_SCHEMA,
        "run_id": run.run_id,
        "task_id": run.task_id,
        "method_id": run.method_id,
        "launch_seq": record.launch_seq,
        "launch_uuid": record.launch_uuid,
        "candidate_id": record.candidate_id,
        "status": record.status,
        "witness": record.witness,
        "charged": record.charged,
        "charge": record.charge.decision,
        "charge_classification": record.charge.classification,
        "metrics": record.metrics,
        "extraction_errors": record.extraction_errors,
        "logs": {"stdout": str(record.stdout_path), "stderr": str(record.outcome.stderr_path)},
        "benchmark_verdict_deferred": True,
    }


def _check_admission(task: Any, views: list[Any], candidate_ids: list[str]) -> None:
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ProtocolError("a batch may not repeat a candidate_id")
    existing = {view.candidate_id for view in views}
    for candidate_id in candidate_ids:
        if candidate_id in existing:
            raise ProtocolError(
                f"candidate {candidate_id!r} already has a ledger intent. One run per "
                "candidate means failures and unresolved launches are not bought again"
            )
    references = [view for view in views if view.is_reference_launch and view.resolved]
    if not references:
        raise ProtocolError("the reference launch must resolve before any candidate launch")
    expected = task.reference_launch.get("assert_exact") or {}
    mismatches = [
        name for name, wanted in expected.items()
        if references[0].metrics.get(name) is None
        or float(references[0].metrics[name]) != float(wanted)
    ]
    if mismatches:
        raise ProtocolError("the reference launch does not reproduce: " + ", ".join(mismatches))
    spent = sum(1 for view in views if view.charge().charged)
    if len(candidate_ids) > task.max_launches - spent:
        raise ProtocolError(
            f"budget exhausted: {spent} charged launches against cap {task.max_launches}; "
            f"{len(candidate_ids)} candidates requested"
        )


@contextmanager
def _exclusive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
