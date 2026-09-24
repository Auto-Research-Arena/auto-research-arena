"""Freeze and verify the implementation repository behind a method definition."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable


SCHEMA = "autoarena/method-source/1"


class SourceError(Exception):
    """A method source cannot be identified reproducibly."""


def lock(directory: Path) -> Dict[str, str]:
    """Record the supplied method directory's content identity."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise SourceError(f"method source is not a directory: {directory}")
    return {"schema": SCHEMA, "kind": "directory", "sha256": _directory_digest(directory)}


def verify(expected: Dict[str, str], directory: Path) -> Dict[str, str]:
    """Verify the initial private copy before running method setup."""
    if not isinstance(expected, dict) or expected.get("schema") != SCHEMA or expected.get("kind") != "directory":
        raise SourceError("unsupported method-source record")
    observed = lock(directory)
    if expected != observed:
        raise SourceError("method source differs from its recorded content identity")
    return observed


def _files(root: Path) -> Iterable[Path]:
    for current, directories, files in os.walk(root):
        for name in directories:
            # A symlinked directory is the escape case and stays refused. os.walk does not
            # descend into one, so admitting it would drop its whole subtree from the digest
            # silently -- two different trees would lock to the same identity.
            if name in {".git", ".venv", "__pycache__"}:
                continue
            if (Path(current) / name).is_symlink():
                raise SourceError(
                    f"method source contains a symlinked directory: {Path(current) / name}"
                )
        directories[:] = sorted(
            name for name in directories if name not in {".git", ".venv", "__pycache__"}
        )
        for name in sorted(files):
            if name not in {".git", ".venv", "__pycache__"}:
                yield Path(current) / name


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        # copytree materializes file links, so identify the same bytes it copies.
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
