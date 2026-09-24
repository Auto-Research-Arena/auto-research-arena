"""Apply the small runtime patch in the engine's private setup copy and install it."""
from pathlib import Path
import hashlib
import subprocess
import shutil
import sys

ROOT = Path(__file__).resolve().parent
RUNNER_SHA256 = "1f1feebf34e7e1346f6ab39c62827d31b6b60a0305f88d7ce050e4d5bb5414f5"


def main():
    upstream = ROOT / "upstream"
    runner = upstream / "autoresearch_automl/core/runner.py"
    if hashlib.sha256(runner.read_bytes()).hexdigest() != RUNNER_SHA256:
        raise RuntimeError("TPE Runner differs from the pinned upstream source")
    subprocess.run(["git", "apply", "--unsafe-paths", str(ROOT / "runtime.patch")],
                   cwd=upstream, check=True)
    subprocess.run([shutil.which("uv"), "pip", "install", "--python", sys.executable, "--no-deps", str(upstream)], check=True)


if __name__ == "__main__":
    main()
