"""Run one local subprocess on a fixed GPU slice and preserve its outcome."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .backend import LaunchOutcome, LaunchRequest, Worker, now

# Grace between SIGTERM and SIGKILL on a timeout. Long enough for a training script to
# flush its stdout -- the log is the measurement, so losing it to an impatient SIGKILL
# turns a timeout into an unreadable launch.
TERM_GRACE_SECONDS = 30


class SubprocessWorker(Worker):
    """A serial worker on this host with fixed device visibility."""

    def __init__(
        self,
        worker_id: str,
        gpu_ids: List[str],
        compute_run_id: str,
        *,
        env: Optional[Mapping[str, str]] = None,
        mounts: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(worker_id, gpu_ids, compute_run_id)
        self.base_env = dict(env or {})
        self.mounts = dict(mounts or {})

    def run(self, request: LaunchRequest) -> LaunchOutcome:
        for path in (request.stdout_path, request.stderr_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.update(self.base_env)
        env.update(request.env)
        # Operator allocation and launch identity remain authoritative.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpu_ids)
        env["AUTOARENA_WORKER_ID"] = self.worker_id
        env["AUTOARENA_LAUNCH_ID"] = request.launch_id

        from engine.compute.storage import mount_command
        command = mount_command(request.command, self.mounts)
        started_at, started = now(), time.monotonic()
        timed_out = False
        notes = ""

        with open(request.stdout_path, "wb") as out, open(request.stderr_path, "wb") as err:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=request.cwd,
                    env=env,
                    stdout=out,
                    stderr=err,
                    # Its own process group, so a timeout kills the training script's
                    # children too. Without this, a killed launcher leaves a python
                    # process holding the GPU and the next launch OOMs for no visible
                    # reason -- which the benchmark would record as a real measurement.
                    start_new_session=True,
                )
            except (OSError, ValueError) as error:
                return self.refused(request, f"could not start command: {error}")

            deadline = None if request.timeout_seconds is None else time.monotonic() + request.timeout_seconds
            try:
                # Leave the leader unreaped so its process-group ID stays reserved
                # if timeout or interruption requires killing remaining children.
                while not _exited(process):
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(.01)
            except KeyboardInterrupt:
                _terminate(process)
                raise
            if timed_out:
                notes = f"timed out after {request.timeout_seconds}s; process group terminated"
                exit_code = _terminate(process)
            else:
                exit_code = process.wait()

        return LaunchOutcome(
            exit_code=exit_code,
            started_at=started_at,
            finished_at=now(),
            wall_time_seconds=round(time.monotonic() - started, 3),
            stdout_path=request.stdout_path,
            stderr_path=request.stderr_path,
            worker_id=self.worker_id,
            compute_run_id=self.compute_run_id,
            timed_out=timed_out,
            notes=notes,
        )


def _exited(process: subprocess.Popen) -> bool:
    """Observe direct-child exit without releasing its process-group identity."""
    return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None


def _terminate(process: subprocess.Popen) -> Optional[int]:
    """SIGTERM the group, then SIGKILL. Returns the exit code, or None if never reaped."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while not _exited(process) and time.monotonic() < deadline:
        time.sleep(.01)
    # The leader may have exited while a TERM-ignoring child remains. Always
    # escalate before reaping the leader, while its group ID cannot be reused.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        return process.wait(timeout=TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        return None
