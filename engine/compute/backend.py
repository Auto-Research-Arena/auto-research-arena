"""Local GPU workers and the small execution interface used by the benchmark."""

from __future__ import annotations

import importlib
import os
import socket
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import probe


class ComputeError(Exception):
    """Execution resources are unavailable; a command failure is an outcome."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class LaunchRequest:
    """One command to run. Everything the backend is allowed to know about it."""

    launch_id: str
    command: List[str]
    cwd: str
    stdout_path: str
    stderr_path: str
    env: Dict[str, str] = field(default_factory=dict)
    gpus: int = 1
    timeout_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.command or not all(isinstance(part, str) and part for part in self.command):
            raise ComputeError("command must be a non-empty list of non-empty strings")
        for name in ("cwd", "stdout_path", "stderr_path"):
            value = getattr(self, name)
            if not os.path.isabs(value):
                raise ComputeError(f"{name} must be an absolute path, got {value!r}")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ComputeError("timeout_seconds must be positive when set")


@dataclass(frozen=True)
class LaunchOutcome:
    """Observed execution facts; the evaluator decides measurement and accounting."""

    exit_code: Optional[int]
    started_at: str
    finished_at: str
    wall_time_seconds: float
    stdout_path: str
    stderr_path: str
    worker_id: str
    compute_run_id: str
    preempted: bool = False
    timed_out: bool = False
    # No exit code does not by itself mean that execution never started.
    launched: bool = True
    notes: str = ""


class Worker(ABC):
    """One serialised stream of launches, holding its own GPUs.

    The harness never calls `run` concurrently on the same worker, so an implementation
    does not need to be reentrant. A worker is what the benchmark calls a *lane*, and its
    identity is recorded per launch because co-tenancy measurably affects results.
    """

    def __init__(self, worker_id: str, gpu_ids: List[str], compute_run_id: str) -> None:
        self.worker_id = worker_id
        self.gpu_ids = list(gpu_ids)
        self.compute_run_id = compute_run_id

    @abstractmethod
    def run(self, request: LaunchRequest) -> LaunchOutcome:
        """Run one command to completion and report. Never raise for a failed command."""

    def prepare_node(self) -> None:
        """Optional setup, called once per new worker after all workers are created."""

    def refused(self, request: LaunchRequest, reason: str) -> LaunchOutcome:
        """Record a command that never started, including its own diagnostic logs."""
        stamp = now()
        for path in (request.stdout_path, request.stderr_path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        # stdout must exist even here: the charging rule reads it for the witness, and an
        # absent file is not the same fact as an empty one.
        open(request.stdout_path, "a", encoding="utf-8").close()
        with open(request.stderr_path, "a", encoding="utf-8") as handle:
            handle.write(reason + "\n")
        return LaunchOutcome(
            exit_code=None,
            started_at=stamp,
            finished_at=stamp,
            wall_time_seconds=0.0,
            stdout_path=request.stdout_path,
            stderr_path=request.stderr_path,
            worker_id=self.worker_id,
            compute_run_id=self.compute_run_id,
            launched=False,
            notes=reason,
        )


CONFIG_KEYS = frozenset({"backend", "accelerator", "visible_devices", "run_id", "mounts", "env"})


class Backend:
    """Run on this host's allocated GPUs. Subclasses may supply a different Worker."""

    extra_config_keys: Tuple[str, ...] = ()

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})
        unknown = set(self.config) - CONFIG_KEYS - set(self.extra_config_keys)
        if unknown:
            raise ComputeError("unknown keys in the compute config: " + ", ".join(sorted(unknown)))
        self.declared_accelerator = self.config.get("accelerator")
        self.visible_devices = _normalize_visible(self.config.get("visible_devices"))
        self.mounts = dict(self.config.get("mounts") or {})
        self.env = dict(self.config.get("env") or {})
        self.compute_run_id = self.config.get("run_id") or f"local-{socket.gethostname()}-{uuid.uuid4().hex[:12]}"

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, Mapping):
            raise ComputeError("compute config must be an object")
        return cls(config)

    def describe(self):
        gpu_ids = _gpu_pool(self.visible_devices)
        names = self.gpu_names(gpu_ids)
        return {
            "backend": self.config.get("backend", "local"),
            "host": socket.gethostname(),
            "accelerator": names[0] if names and len(set(names)) == 1 else None,
            "gpu_names": names, "compute_run_id": self.compute_run_id,
            "visible_devices": ",".join(gpu_ids),
        }

    def gpu_names(self, gpu_ids):
        """Observe selected IDs in order; [] means the full selection is unknown."""
        return probe.gpu_names(gpu_ids)

    def preflight(self):
        problems = []
        gpu_ids = _gpu_pool(self.visible_devices)
        names = self.gpu_names(gpu_ids)
        if not gpu_ids:
            problems.append("no GPUs selected")
        elif len(names) != len(gpu_ids):
            problems.append("could not observe all selected GPU IDs: " + ", ".join(gpu_ids))
        if self.declared_accelerator and names and not all(
            probe.matches(self.declared_accelerator, name) for name in names
        ):
            problems.append(f"config declares accelerator {self.declared_accelerator!r} but the visible GPUs are "
                            + ", ".join(sorted(set(names))))
        return problems

    def workers(self, lanes, gpus_per_lane):
        if lanes < 1 or gpus_per_lane < 1:
            raise ComputeError("lanes and gpus_per_lane must both be at least 1")
        gpu_ids = resolve_gpu_ids(lanes * gpus_per_lane, visible=self.visible_devices)
        workers = [self.make_worker(index, gpu_ids[index * gpus_per_lane:(index + 1) * gpus_per_lane])
                   for index in range(lanes)]
        for worker in workers:
            worker.prepare_node()
        return workers

    def make_worker(self, index, gpu_ids):
        from .runner import SubprocessWorker
        return SubprocessWorker(f"lane{index + 1}", gpu_ids, self.compute_run_id,
                                mounts=self.mounts, env=self.env)

    def release(self):
        """Local allocations remain operator-owned."""


def load_backend(config: Mapping[str, Any]) -> Backend:
    """Load local execution or an explicitly supplied Worker backend."""
    if not isinstance(config, Mapping):
        raise ComputeError("compute config must be an object")
    spec = config.get("backend")
    if not isinstance(spec, str) or not spec:
        raise ComputeError(
            "compute.backend is required: 'local' or a 'module:Class' path"
        )
    if ":" in spec:
        module_name, _, class_name = spec.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise ComputeError(f"cannot import backend module {module_name!r}: {error}") from error
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ComputeError(f"{module_name!r} has no attribute {class_name!r}")
        if not (isinstance(cls, type) and issubclass(cls, Backend)):
            raise ComputeError(f"{spec!r} does not subclass compute.backend.Backend")
        return cls.from_config(config)
    if spec == "local":
        return Backend.from_config(config)
    raise ComputeError(
        f"unknown backend {spec!r}; use 'local' or 'module:Class'. "
        "A third-party backend is named as 'module:Class'"
    )


def _normalize_visible(value: Any) -> Optional[str]:
    """Normalize a list or CUDA mask, preserving an explicitly empty selection."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item).strip() for item in value)
    if not isinstance(value, str):
        raise ComputeError(
            "visible_devices must be a list of device ids or a comma-separated string, got "
            + type(value).__name__
        )
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if len(ids) != len(set(ids)):
        raise ComputeError("visible_devices contains duplicate GPU IDs")
    return ",".join(ids)


def _gpu_pool(visible):
    if visible is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = _normalize_visible(visible)
    if visible is None:
        return probe.gpu_ids()
    return visible.split(",") if visible else []


def resolve_gpu_ids(count: int, offset: int = 0, visible: Any = None) -> List[str]:
    """Select a disjoint slice of the explicit mask, environment mask or host GPUs."""
    pool = _gpu_pool(visible)
    slice_ = pool[offset : offset + count]
    if len(slice_) < count:
        raise ComputeError(
            f"need {count} GPU(s) at offset {offset} but only {len(pool)} are visible "
            f"({','.join(pool) or 'empty selection'})"
        )
    return slice_
