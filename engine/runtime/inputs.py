"""Capture the files used to identify a benchmark run and its measurements."""

from pathlib import Path

class InputError(ValueError):
    pass

def read_captured_file(path):
    """Read a regular input file; callers hash the captured bytes where needed."""
    path = Path(path)
    try:
        if not path.is_file():
            raise ValueError("not a regular file")
        return path.read_bytes()
    except (OSError, ValueError) as error:
        raise InputError(f"cannot read input file {path}: {error}") from error
