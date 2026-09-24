"""Local GPU execution and the request/outcome boundary."""

from .backend import (
    Backend,
    ComputeError,
    LaunchOutcome,
    LaunchRequest,
    Worker,
    load_backend,
    resolve_gpu_ids,
)
from .runner import SubprocessWorker

__all__ = [
    "Backend",
    "ComputeError",
    "LaunchOutcome",
    "LaunchRequest",
    "SubprocessWorker",
    "Worker",
    "load_backend",
    "resolve_gpu_ids",
]
