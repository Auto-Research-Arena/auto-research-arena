"""Bounded local GPU observations; unavailable hardware is reported as unknown."""

from __future__ import annotations

import subprocess
from typing import List

QUERY_TIMEOUT_SECONDS = 10


def _query(*fields: str, gpu_ids=None) -> List[str]:
    command = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"]
    if gpu_ids is not None:
        if not gpu_ids:
            return []
        command += ["--id=" + ",".join(gpu_ids)]
    try:
        probe = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if probe.returncode != 0:
        return []
    return [line.strip() for line in probe.stdout.splitlines() if line.strip()]


def gpu_names(gpu_ids) -> List[str]:
    """Observe a complete selection in requested order, resolving index/UUID aliases."""
    rows = [tuple(part.strip() for part in line.split(",", 2))
            for line in _query("index", "uuid", "name", gpu_ids=gpu_ids)]
    if any(len(row) != 3 or not all(row) for row in rows):
        return []
    names, seen = [], set()
    for identity in gpu_ids:
        matching = [row for row in rows if row[0] == identity
                    or (identity.startswith("GPU-") and row[1].startswith(identity))]
        if len(matching) != 1 or matching[0][1] in seen:
            return []
        _, uuid, name = matching[0]
        seen.add(uuid)
        names.append(name)
    return names


def gpu_ids() -> List[str]:
    """Observed device indices, without assuming contiguous numbering."""
    return _query("index")


def matches(declared: str, reported: str) -> bool:
    """Match a label such as NVIDIA_A100_80GB to NVIDIA A100-SXM4-80GB."""
    squash = "".join(ch for ch in reported.upper() if ch.isalnum())
    tokens = [
        "".join(ch for ch in part if ch.isalnum())
        for part in declared.upper().replace("-", "_").split("_")
    ]
    tokens = [token for token in tokens if token]
    return bool(tokens) and all(token in squash for token in tokens)
