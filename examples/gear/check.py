"""Check the pinned native controller and the installed execution client."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
settings = json.loads((root / "settings.json").read_text())
if hashlib.sha256((root / "upstream/gear.py").read_bytes()).hexdigest() != settings["gear_sha256"]:
    raise RuntimeError("Pinned GEAR controller changed")
from autoarena import Benchmark
subprocess.run(["claude", "--version"], check=True)
