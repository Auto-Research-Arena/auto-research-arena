"""Optional CPU-command conformance checks for a compute backend.

    python3 -m engine.compute.selfcheck --config my-config.json
    python3 -m engine.compute.selfcheck --backend local --backend-config '{"visible_devices":[0,1]}'

Checks command success, failure, timeout, refusal logs and distinct worker allocations.
The second command supplies synthetic device IDs for a CPU-only check; it does not
establish hardware readiness. A benchmark run still requires real selected GPUs.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .backend import Backend, ComputeError, LaunchRequest, load_backend


def _request(root: Path, name: str, command: List[str], **kwargs: Any) -> LaunchRequest:
    return LaunchRequest(
        launch_id=name,
        command=command,
        cwd=str(root),
        stdout_path=str(root / f"{name}.stdout"),
        stderr_path=str(root / f"{name}.stderr"),
        **kwargs,
    )


def run_checks(backend: Backend, lanes: int = 2) -> Tuple[List[str], Dict[str, Any]]:
    """Return (failures, notes). An empty failure list means the backend conforms."""
    failures: List[str] = []
    notes: Dict[str, Any] = {"describe": backend.describe(), "preflight": backend.preflight()}

    description = notes["describe"]
    if not isinstance(description, dict) or not description:
        failures.append("describe() must return a non-empty dict of provenance")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            workers = backend.workers(lanes, 1)
            if len(workers) != lanes:
                failures.append(f"workers({lanes}, 1) returned {len(workers)} worker(s)")
                return failures, notes
            if len({worker.worker_id for worker in workers}) != len(workers):
                failures.append("workers must have distinct worker_ids; lane identity is recorded")
            gpu_sets = [tuple(worker.gpu_ids) for worker in workers]
            if any(len(ids) != 1 for ids in gpu_sets):
                failures.append("each worker must hold the requested one GPU")
            gpu_ids = [identity for ids in gpu_sets for identity in ids]
            if len(set(gpu_ids)) != len(gpu_ids):
                failures.append(
                    f"lanes share GPUs {gpu_sets}: two lanes on one device serialise "
                    "invisibly and both launches report a wall clock that includes the other"
                )
            run_ids = {worker.compute_run_id for worker in workers}
            if len(run_ids) != 1 or not next(iter(run_ids)):
                failures.append(
                    f"compute_run_id must be one non-empty value per allocation, got {run_ids}"
                )
            notes["compute_run_id"] = sorted(run_ids)

            worker = workers[0]

            # 1. success
            outcome = worker.run(_request(root, "ok", [sys.executable, "-c", "print('hello')"]))
            if outcome.exit_code != 0:
                failures.append(f"a successful command reported exit_code {outcome.exit_code}")
            if not Path(outcome.stdout_path).is_file():
                failures.append("stdout_path is not a real file after a successful launch")
            elif "hello" not in Path(outcome.stdout_path).read_text():
                failures.append("stdout was not captured: the log IS the measurement")
            if not outcome.launched:
                failures.append("a command that ran reported launched=False")

            # 2. failure
            outcome = worker.run(
                _request(root, "fail", [sys.executable, "-c", "import sys; print('x'); sys.exit(3)"])
            )
            if outcome.exit_code != 3:
                failures.append(f"a failing command reported exit_code {outcome.exit_code}, not 3")
            if not Path(outcome.stdout_path).is_file():
                failures.append(
                    "no stdout file after a failed launch. Charging reads the launch's own "
                    "stdout for the GPU-work witness, so a missing file makes a crash "
                    "indistinguishable from a launch that never started -- and they charge "
                    "differently"
                )

            # 3. timeout
            outcome = worker.run(
                _request(
                    root,
                    "hang",
                    [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(120)"],
                    timeout_seconds=3,
                )
            )
            if not outcome.timed_out:
                failures.append("a command exceeding timeout_seconds did not report timed_out")
            if outcome.wall_time_seconds > 90:
                failures.append(
                    f"timeout was not enforced: the launch ran {outcome.wall_time_seconds}s "
                    "against a 3s limit. A hung launch that is never killed ends the run "
                    "silently"
                )

            # 4. command-start refusal
            outcome = worker.run(
                _request(
                    root,
                    "unstartable",
                    [str(root / "missing-executable")],
                )
            )
            if outcome.launched:
                failures.append(
                    "an unstartable command reported launched=True"
                )
            if outcome.exit_code is not None:
                failures.append(
                    f"a refused launch reported exit_code {outcome.exit_code}; it must be "
                    "None, because 'no exit code and no GPU work' is what makes it refundable"
                )
            if not Path(outcome.stdout_path).is_file():
                failures.append("a refused launch left no stdout file")
            if not Path(outcome.stderr_path).is_file():
                failures.append("a refused launch left no stderr file")
        finally:
            backend.release()

    return failures, notes


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, help="run config JSON containing a `compute` object")
    parser.add_argument("--backend", help="local or module:Class, instead of --config")
    parser.add_argument(
        "--backend-config",
        default="",
        help="JSON object of backend settings, merged with --backend",
    )
    parser.add_argument("--lanes", type=int, default=2)
    args = parser.parse_args(argv)

    if args.config:
        document = json.loads(args.config.read_text())
        config = document.get("compute", document)
    elif args.backend:
        config = {**json.loads(args.backend_config or "{}"), "backend": args.backend}
    else:
        parser.error("pass --config or --backend")

    try:
        backend = load_backend(config)
        failures, notes = run_checks(backend, lanes=args.lanes)
    except ComputeError as error:
        print(f"compute check error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(notes, indent=2, sort_keys=True, default=str))
    if notes["preflight"]:
        print("\npreflight warnings (not conformance failures):")
        for problem in notes["preflight"]:
            print(f"  ! {problem}")
    if failures:
        print(f"\n{len(failures)} conformance failure(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nCPU command checks passed; see engine/compute/README.md for the execution contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
