"""Durable, one-shot background delivery to the existing candidate batch boundary.

Jobs contain JSON data, never executable paths or shell commands. A detached worker uses
the same lane evaluator as a foreground caller and preserves its raw responses.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from engine import ROOT as _ROOT
from engine.records import runs
from engine.evaluation.protocol import CANDIDATE_ID, ProtocolError, research_payload

SCHEMA = "autoarena/dispatch-job/1"
REQUEST_SCHEMA = "autoarena/dispatch-request/1"
RESPONSE_SCHEMA = "autoarena/dispatch-response/1"
JOB_ID = re.compile(r"[0-9a-f]{32}")
REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
DIGEST = re.compile(r"[0-9a-f]{64}")
OWNERSHIP_PROTOCOL = "autoarena/inherited-handoff/1"


class DispatchError(Exception):
    """A background request cannot be identified or safely delivered."""


def submit(run_path: Path, candidates: list, *, request_id: str) -> dict[str, Any]:
    """Persist one request and start a worker that survives this caller's exit.

    Submission does not reserve a second accounting system or bypass launch checks.
    Concurrent submissions for an existing candidate are refused by the lane ledger.
    A stable request_id rejoins exactly the same request, never its worker.
    Candidate source directories must remain immutable after submission.
    """
    run_path = Path(run_path).resolve()
    job_id = _request_job_id(request_id)
    run = runs.open_run(run_path)
    _run_binding(run_path)
    candidates = _candidate_data(candidates)
    root = run_path / "method/dispatch"
    if root.is_symlink() or not root.resolve().is_relative_to(run_path):
        raise DispatchError("dispatch storage must stay inside this run")
    root.mkdir(parents=True, exist_ok=True)
    lease_path = root / (".request-" + job_id + ".lock")
    if lease_path.is_symlink():
        raise DispatchError("request identity lock cannot be a symbolic link")
    with lease_path.open("a+b") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DispatchError("request is being published; retry the same request_id") from error
        if (root / job_id).exists():
            _, _, _, request = _load_job(run_path, job_id)
            if (request.get("request_id") != request_id
                    or _encoded(request["candidates"]) != _encoded(candidates)
                    or request["binding"] != _run_binding(run_path)):
                raise DispatchError("request_id already belongs to different inputs")
            return status(run_path, job_id)
        return _submit_new(run_path, run, root, candidates, job_id, request_id)


def _request_job_id(request_id: str) -> str:
    if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
        raise DispatchError("request_id must be 1–128 non-secret identifier characters")
    return hashlib.sha256(("autoarena/dispatch-key/1\0" + request_id).encode()).hexdigest()[:32]


def lookup(run_path: Path, request_id: str) -> dict[str, Any] | None:
    """Recover a known request's job after a lost submit response; never spawn work.

    An incomplete publication fails closed, not as permission to submit another job.
    """
    run_path = Path(run_path).resolve()
    job_id = _request_job_id(request_id)
    runs.open_run(run_path)
    root = run_path / "method/dispatch"
    if root.is_symlink() or not root.resolve().is_relative_to(run_path):
        raise DispatchError("dispatch storage must stay inside this run")
    if (root / job_id).is_symlink():
        raise DispatchError("dispatch job cannot be a symbolic link")
    if not (root / job_id).exists():
        return None
    _, _, _, request = _load_job(run_path, job_id)
    if (request.get("request_id") != request_id
            or request["binding"] != _run_binding(run_path)):
        raise DispatchError("request_id already belongs to different inputs")
    return status(run_path, job_id)


def wait(run_path: Path, job: dict) -> dict:
    """Wait for the original worker to release its lease, without polling files."""
    if job["status"] in {"completed", "failed", "unknown"}:
        return job
    _, job_dir, _, _ = _load_job(Path(run_path).resolve(), job["job_id"])
    with (job_dir / ".handoff.lock").open("rb") as lease:
        fcntl.flock(lease, fcntl.LOCK_SH)
    return status(run_path, job["job_id"])


def _submit_new(run_path, run, root, candidates, job_id, request_id):
    job_dir = root / job_id
    job_dir.mkdir()  # A collision is refused, never an instruction to reuse a job.
    request = {"schema": REQUEST_SCHEMA, "job_id": job_id, "request_id": request_id,
               "run_id": run.run_id, "run_path": str(run_path),
               "binding": _run_binding(run_path), "candidates": candidates}
    payload = _encoded(request)
    digest = hashlib.sha256(payload).hexdigest()
    metadata = {"schema": SCHEMA, "job_id": job_id, "run_id": run.run_id,
                "run_path": str(run_path), "created_at": runs.utc_now(),
                "request_sha256": digest}
    submitter = _process_identity(os.getpid())
    if submitter is None:
        raise DispatchError("cannot establish the submitting process identity")
    metadata["ownership"] = {"protocol": OWNERSHIP_PROTOCOL, "submitter": {
        "hostname": os.uname().nodename, "boot_id": _boot_id(), **submitter,
    }}
    # The lease exists before the committed job and is inherited by the child.
    # A parent exit cannot leave an unleased post-spawn/pre-identity gap.
    with (job_dir / ".handoff.lock").open("x+b") as handoff:
        fcntl.flock(handoff, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _create_file(job_dir / "request.json", payload)
        _create_file(job_dir / "job.json", _encoded(metadata))
        process = None
        try:
            with (job_dir / "stdout.log").open("x") as stdout, \
                    (job_dir / "stderr.log").open("x") as stderr:
                from engine.runtime import lifecycle as runtime, environment
                _, definition = runtime.open_run(run_path)
                process = subprocess.Popen(
                    [sys.executable, "-m", "engine.runtime.dispatch", "--run", str(run_path),
                     "--job-id", job_id, "--request-sha256", digest,
                     "--handoff-fd", str(handoff.fileno())],
                    cwd=_ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    env=environment.runner_environment(definition),
                    start_new_session=True, pass_fds=(handoff.fileno(),),
                )
                identity = _process_identity(process.pid)
                _create_file(job_dir / "submitted-worker.json", _encoded({
                    "job_id": job_id, "request_sha256": digest,
                    "hostname": os.uname().nodename, "boot_id": _boot_id(),
                    "identity_recorded": identity is not None,
                    **(identity or {"pid": process.pid}),
                }))
                threading.Thread(target=process.wait, daemon=True).start()
        except OSError as error:
            if process is not None:
                # A spawned child still owns its inherited lease. A failed parent
                # record write does not prove that child stopped or that no work ran.
                raise DispatchError(
                    f"job {job_id} spawned its worker but could not publish ownership; "
                    "inspect this job without resubmitting it"
                ) from error
            _terminal(job_dir, metadata, error=error)
        return status(run_path, job_id)


def status(run_path: Path, job_id: str) -> dict[str, Any]:
    """Read transport state, including unknown ownership, without a verdict."""
    run_path, job_dir, metadata, _ = _load_job(run_path, job_id)
    terminal = _response_status(job_dir, metadata)
    if terminal is not None:
        return terminal
    result = {**metadata, "status": "pending", "benchmark_verdict_deferred": True}
    lost_message = None
    unknown = False
    started_path = job_dir / "started.json"
    if started_path.is_file():
        started = _read_json(started_path)
        if (started.get("job_id") != job_id
                or started.get("request_sha256") != metadata["request_sha256"]):
            raise DispatchError("worker identity does not belong to this request")
        result.update(status="running", started_at=started["started_at"])
        if not _worker_alive(started):
            lost_message = ("worker ended without a terminal response; inspect this run's "
                            "ledger. This request will not be retried")
    elif (job_dir / "submitted-worker.json").is_file():
        submitted = _submitted_identity(job_dir, metadata)
        if not submitted.get("identity_recorded"):
            unknown = True
        elif not _worker_alive(submitted):
            lost_message = ("submitted worker exited or could not be identified before "
                            "starting the request. This request will not be retried")
    else:
        unknown = True
    ownership = metadata["ownership"]
    if unknown or lost_message is not None:
        held = _handoff_held(job_dir)
        if held:
            # This includes a live child blocked on publishing its identity. Neither
            # elapsed time nor a missing final JSON name makes that child dead.
            result.update(status="pending", ownership_state="handoff_lease_held")
            return result
        if held is False and unknown and not _worker_alive(ownership["submitter"]):
            # Every submitter and worker keeps this lease. With a
            # proven dead submitter and a free lease, no original worker can remain.
            lost_message = ("submission ownership ended before a worker identity was "
                            "published; no replay is authorized")
            unknown = False
    if lost_message is not None:
        # The worker publishes its response before exiting. It may have done both
        # after our first file check, so process death alone cannot imply a lost result.
        terminal = _response_status(job_dir, metadata)
        if terminal is not None:
            return terminal
        result.update(status="failed", error={"type": "WorkerLost", "message": lost_message})
    elif unknown:
        # Identity or terminal publication may have raced with the first observation.
        terminal = _response_status(job_dir, metadata)
        if terminal is not None:
            return terminal
        if ((job_dir / "started.json").is_file()
                or ((job_dir / "submitted-worker.json").is_file()
                    and _submitted_identity(job_dir, metadata).get("identity_recorded"))):
            return status(run_path, job_id)
        result.update(status="unknown", error={
            "type": "OwnershipUnknown",
            "message": "No published original-worker identity can establish ownership. "
                       "The worker may still exist; this is not a failed measurement. "
                       "Do not resubmit or replay this request. Recover its ownership records.",
        })
    return result


def _response_status(job_dir: Path, metadata: dict) -> dict[str, Any] | None:
    response_path = job_dir / "response.json"
    if not response_path.is_file():
        return None
    response = _read_json(response_path)
    if (response.get("schema") != RESPONSE_SCHEMA
            or response.get("job_id") != metadata["job_id"]
            or response.get("run_id") != metadata["run_id"]
            or response.get("request_sha256") != metadata["request_sha256"]
            or response.get("status") not in {"completed", "failed"}):
        raise DispatchError("dispatch response does not belong to this request")
    return {**metadata, "benchmark_verdict_deferred": True,
            **{key: value for key, value in response.items() if key != "schema"}}


def _work(run_path: Path, job_id: str, expected_digest: str,
          handoff_fd: int | None = None) -> None:
    if not isinstance(expected_digest, str) or not DIGEST.fullmatch(expected_digest):
        raise DispatchError("invalid request digest")
    run_path, job_dir, metadata, request = _load_job(run_path, job_id)
    if metadata["request_sha256"] != expected_digest:
        raise DispatchError("worker invocation does not match the submitted request")
    with (job_dir / ".worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DispatchError("this dispatch job already has a worker") from error
        if (job_dir / "started.json").exists() or (job_dir / "response.json").exists():
            raise DispatchError("this dispatch job was already attempted; it cannot be replayed")
        _check_handoff_fd(job_dir, handoff_fd)
        identity = _process_identity(os.getpid())
        if identity is None:
            raise DispatchError("cannot establish the background worker's identity")
        if (job_dir / "submitted-worker.json").is_file():
            submitted = _submitted_identity(job_dir, metadata)
            if (not submitted.get("identity_recorded")
                    or submitted.get("hostname") != os.uname().nodename
                    or submitted.get("boot_id") != _boot_id()
                    or submitted.get("pid") != identity["pid"]
                    or submitted.get("start_ticks") != identity["start_ticks"]):
                raise DispatchError("this request belongs to its original submitted worker; no replay")
        _create_file(job_dir / "started.json", _encoded({
            "job_id": job_id, "request_sha256": expected_digest,
            "started_at": runs.utc_now(), "hostname": os.uname().nodename,
            "boot_id": _boot_id(), **identity,
        }))
        try:
            if request["binding"] != _run_binding(run_path):
                raise DispatchError("run or lane binding changed after background submission")
            from engine.runtime import lifecycle as runtime
            results = runtime.evaluate_batch(run_path, request["candidates"])
        except Exception as error:
            _terminal(job_dir, metadata, error=error)
        else:
            _terminal(job_dir, metadata, results=results)


def _terminal(job_dir: Path, metadata: dict, *, results=None, error=None) -> None:
    response = {"schema": RESPONSE_SCHEMA, "job_id": metadata["job_id"],
                "run_id": metadata["run_id"], "request_sha256": metadata["request_sha256"],
                "finished_at": runs.utc_now(), "status": "failed" if error is not None else "completed",
                "benchmark_verdict_deferred": True}
    if error is not None:
        response["error"] = {"type": type(error).__name__, "message": str(error)}
    else:
        response["results"] = results
    # Match foreground evaluation/ledger serialization, including a recorded NaN/Inf
    # from a diverged launch. Requests stay strict; observed values are never rewritten.
    payload = (json.dumps(response, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _create_file(job_dir / "response.json", payload)


def _load_job(run_path: Path, job_id: str):
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise DispatchError("job_id must be the 32-character hexadecimal submission id")
    run_path = Path(run_path).resolve()
    run = runs.open_run(run_path)
    root = run_path / "method/dispatch"
    job_dir = root / job_id
    if (root.is_symlink() or not job_dir.is_dir() or job_dir.is_symlink()
            or not job_dir.resolve().is_relative_to(run_path)):
        raise DispatchError("dispatch job does not exist inside this run")
    metadata = _read_json(job_dir / "job.json")
    if (metadata.get("schema") != SCHEMA or metadata.get("job_id") != job_id
            or metadata.get("run_id") != run.run_id
            or metadata.get("run_path") != str(run_path)):
        raise DispatchError("dispatch job belongs to another run or request")
    _ownership(metadata)
    digest = metadata.get("request_sha256")
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise DispatchError("dispatch job has no valid request digest")
    request_path = job_dir / "request.json"
    request = _read_json(request_path)
    if hashlib.sha256(request_path.read_bytes()).hexdigest() != digest:
        raise DispatchError("background request changed after submission")
    if (request.get("schema") != REQUEST_SCHEMA or request.get("job_id") != job_id
            or request.get("run_id") != run.run_id or request.get("run_path") != str(run_path)):
        raise DispatchError("background request belongs to another run or job")
    if not isinstance(request.get("binding"), dict):
        raise DispatchError("background request has no run binding")
    if "request_id" in request and _request_job_id(request["request_id"]) != job_id:
        raise DispatchError("background request identity does not match its job")
    _candidate_data(request.get("candidates"))
    return run_path, job_dir, metadata, request


def _candidate_data(candidates: list) -> list:
    if not isinstance(candidates, list) or not candidates:
        raise DispatchError("background candidates must be a nonempty list")
    normalized = []
    for item in candidates:
        if (not isinstance(item, Mapping) or not {"candidate_id", "source"} <= set(item)
                or set(item) - {"candidate_id", "source", "parent_id", "idea", "research_log"}):
            raise DispatchError("each candidate needs candidate_id, source and research_log; parent_id and idea are optional")
        item = {"parent_id": None, "idea": {}, **item}
        try:
            item["research_log"] = research_payload(item.get("research_log"))
        except ProtocolError as error:
            raise DispatchError(str(error)) from error
        if not isinstance(item["candidate_id"], str) or not CANDIDATE_ID.fullmatch(item["candidate_id"]):
            raise DispatchError("candidate candidate_id is not a valid identifier")
        parent = item["parent_id"]
        if parent is not None and (not isinstance(parent, str) or not parent.strip()):
            raise DispatchError("candidate parent_id must be a nonblank string when supplied")
        if not isinstance(item["source"], (str, Path)) or not str(item["source"]):
            raise DispatchError("candidate source must be a directory path")
        source_path = Path(item["source"]).absolute()
        if any(path.is_symlink() for path in (source_path, *source_path.parents)):
            raise DispatchError("candidate source path must not contain symbolic links")
        if not isinstance(item["idea"], Mapping):
            raise DispatchError("candidate idea must be a JSON object")
        normalized.append({**item, "source": str(Path(item["source"]).resolve()),
                           "idea": dict(item["idea"])})
    try:
        return json.loads(_encoded(normalized))
    except (TypeError, ValueError) as error:
        raise DispatchError("background candidates must contain JSON-serializable data") from error


def _run_binding(run_path: Path) -> dict[str, str]:
    from engine.runtime import lifecycle as runtime
    return runtime.binding(run_path)


def _ownership(metadata: dict) -> dict:
    ownership = metadata.get("ownership")
    if (not isinstance(ownership, dict)
            or ownership.get("protocol") != OWNERSHIP_PROTOCOL
            or not isinstance(ownership.get("submitter"), dict)):
        raise DispatchError("unsupported dispatch ownership protocol")
    submitter = ownership["submitter"]
    if (not isinstance(submitter.get("hostname"), str) or not submitter["hostname"]
            or not isinstance(submitter.get("boot_id"), str) or not submitter["boot_id"]
            or type(submitter.get("pid")) is not int or submitter["pid"] < 1
            or type(submitter.get("start_ticks")) is not int or submitter["start_ticks"] < 0):
        raise DispatchError("dispatch submitter has no valid process identity")
    return ownership


def _handoff_held(job_dir: Path) -> bool | None:
    """Observe a lease; never infer death merely from a missing ownership file."""
    path = job_dir / ".handoff.lock"
    if path.is_symlink():
        raise DispatchError("dispatch ownership lease may not be a symbolic link")
    if not path.is_file():
        return None
    with path.open("r+b") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        else:
            fcntl.flock(lease, fcntl.LOCK_UN)
            return False


def _check_handoff_fd(job_dir: Path, descriptor: int | None) -> None:
    """Only the original child receives this already-held open-file description."""
    if type(descriptor) is not int or descriptor < 3:
        raise DispatchError("original submitted worker requires its inherited handoff lease; no replay")
    try:
        observed = os.fstat(descriptor)
        expected = (job_dir / ".handoff.lock").stat()
    except OSError as error:
        raise DispatchError("cannot verify the original inherited handoff lease; no replay") from error
    if ((observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino)
            or _handoff_held(job_dir) is not True):
        raise DispatchError("worker did not inherit this job's held handoff lease; no replay")
    try:
        # A separately opened descriptor for the same inode cannot join the lock.
        # The inherited open-file description already owns it, so this succeeds
        # without releasing or transferring the original lease.
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise DispatchError("worker descriptor is not the inherited handoff lease; no replay") from error
    # The worker retains the lease through terminal publication and process exit,
    # but training subprocesses must not keep it alive after their worker ends.
    os.set_inheritable(descriptor, False)


def _submitted_identity(job_dir: Path, metadata: dict) -> dict:
    submitted = _read_json(job_dir / "submitted-worker.json")
    if (submitted.get("job_id") != metadata["job_id"]
            or submitted.get("request_sha256") != metadata["request_sha256"]):
        raise DispatchError("submitted worker identity does not belong to this request")
    return submitted


def _read_json(path: Path) -> dict:
    if path.is_symlink():
        raise DispatchError("dispatch records may not be symbolic links")
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise DispatchError(f"cannot read dispatch record {path.name}") from error
    if not isinstance(result, dict):
        raise DispatchError(f"dispatch record {path.name} must be a JSON object")
    return result


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _create_file(path: Path, payload: bytes) -> None:
    """Publish a complete record under the caller's writer lock."""
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(payload)
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _process_identity(pid: int) -> dict | None:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise DispatchError(f"cannot inspect background worker identity for PID {pid}") from error
    fields = stat[stat.rfind(")") + 2:].split()
    try:
        return {"pid": pid, "start_ticks": int(fields[19]), "process_state": fields[0]}
    except (IndexError, ValueError) as error:
        raise DispatchError("invalid background worker process identity") from error


def _worker_alive(started: dict) -> bool:
    if started.get("hostname") != os.uname().nodename:
        return True  # Last reported running; this host cannot determine remote liveness.
    if started.get("boot_id") != _boot_id():
        return False
    pid, ticks = started.get("pid"), started.get("start_ticks")
    if not isinstance(pid, int) or pid < 1 or not isinstance(ticks, int):
        raise DispatchError("background job has no valid worker identity")
    observed = _process_identity(pid)
    return bool(observed and observed["start_ticks"] == ticks
                and observed["process_state"] not in {"Z", "X"})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--handoff-fd", type=int)
    args = parser.parse_args(argv)
    try:
        _work(args.run, args.job_id, args.request_sha256, args.handoff_fd)
    except DispatchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
