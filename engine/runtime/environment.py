"""Create method environments before research; never install into GPU runtimes."""

import hashlib
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from engine import ROOT


class EnvironmentError(RuntimeError):
    pass


RUNNER_ENV_KEYS = (
    "PATH", "VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "CONDA_PREFIX",
    "UV_PROJECT_ENVIRONMENT", "UV_PYTHON", "UV_NO_PROJECT", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR",
    "PIP_CACHE_DIR", "PIP_REQUIRE_VIRTUALENV", "XDG_CACHE_HOME", "HF_HOME", "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "TMPDIR",
    "npm_config_cache", "npm_config_prefix",
)


def runner_environment(definition):
    """Do not leak method installation settings into benchmark GPU workers."""
    selected = definition.get("runner_environment", {})
    result = dict(os.environ)
    for key, value in selected.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def validate(recipe):
    if not isinstance(recipe, dict) or set(recipe) != {"setup", "check"}:
        raise EnvironmentError("environment requires only setup and check; scripts select their own runtime")
    if not isinstance(recipe["setup"], list):
        raise EnvironmentError("environment.setup must be a list of command argv arrays (or [])")
    for command in [*recipe["setup"], recipe["check"]]:
        if not isinstance(command, list) or not command or any(
                not isinstance(x, str) or not x or "\0" in x for x in command):
            raise EnvironmentError("setup/check commands must be nonempty argv arrays")


def client_digest():
    return hashlib.sha256((ROOT / "engine/autoarena.py").read_bytes()).hexdigest()


def method_variables(definition, *, setup=False):
    """Expose the client; method scripts select and activate their own runtime."""
    root = Path(definition["environment"]["root"])
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_PYTHON",
                "UV_NO_PROJECT", "PIP_REQUIRE_VIRTUALENV", "CONDA_PREFIX"):
        env.pop(key, None)
    cache = root / "cache"
    env.update(PATH=str(root / "bin") + os.pathsep + env.get("PATH", ""),
               PYTHONPATH=os.pathsep.join([str(root / "client"), str(ROOT)]),
               UV_LINK_MODE="copy", UV_CACHE_DIR=str(cache / "uv"), PIP_CACHE_DIR=str(cache / "pip"),
               PIP_NO_CACHE_DIR="1", UV_PYTHON_INSTALL_DIR=str(root / "python"),
               XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
               TORCH_HOME=str(cache / "torch"), TORCH_EXTENSIONS_DIR=str(cache / "torch_extensions"),
               TRITON_CACHE_DIR=str(cache / "triton"), CUDA_CACHE_PATH=str(cache / "cuda"),
               npm_config_cache=str(cache / "npm"), npm_config_prefix=str(root),
               TMPDIR=str(root / "tmp"), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    if setup:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _run(argv, cwd, env, log, stop_file):
    """Bounded, noninteractive commands; terminate only the group we create."""
    with log.open("xb") as output:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1800
        try:
            while process.poll() is None:
                if stop_file.exists() or time.monotonic() > deadline:
                    raise EnvironmentError(f"setup stopped or exceeded 30 minutes; inspect {log}")
                time.sleep(0.1)
            if process.returncode:
                raise EnvironmentError(f"environment command failed (exit {process.returncode}); inspect {log}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def prepare(run, definition, *, retry_setup=False):
    """Prepare the method, retrying failed setup only when explicitly requested."""
    from engine.runtime import lifecycle as runtime, source
    plan = definition["environment"]
    root = Path(plan["root"])
    receipt = run.path / "engine/environment.json"
    client = root / "client/autoarena.py"
    if retry_setup:
        if run.views() or (run.path / "engine/reference-started.json").exists():
            raise EnvironmentError("setup cannot be retried after measurement has started")
        if receipt.exists():
            raise EnvironmentError("setup already completed; resume without --retry-setup")
    if receipt.exists():
        observed = runtime.read(receipt)
        if (observed.get("plan") != plan or not client.is_file()
                or observed["client_module_sha256"] != hashlib.sha256(client.read_bytes()).hexdigest()):
            raise EnvironmentError("recorded client is missing or changed; no automatic rebuild")
        _check(run, definition, run.path / "engine" / ("environment-check-" + str(time.time_ns()) + ".log"))
        return
    if plan["client_sha256"] != client_digest():
        raise EnvironmentError("client source changed after preparation; start a fresh run")
    logs = run.path / "engine/setup"
    if root.exists():
        if not retry_setup:
            raise EnvironmentError("incomplete setup; inspect its logs, then use resume --retry-setup")
        if not logs.is_dir() or not (root / "source").is_dir() or not client.is_file():
            raise EnvironmentError("initial method copy is incomplete; start a fresh run")
        if client.read_bytes() != (ROOT / "engine/autoarena.py").read_bytes():
            raise EnvironmentError("recorded client changed; restore it before retrying setup")
        logs = logs / ("retry-" + str(time.time_ns()))
    else:
        if logs.exists():
            raise EnvironmentError("private method environment is missing; restore it or start a fresh run")
        root.mkdir(parents=True)
        for folder in ("tmp", "bin", "client"):
            (root / folder).mkdir()
        shutil.copytree(definition["config"]["source"], root / "source",
                        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
        source.verify(definition["source_lock"], root / "source")
        shutil.copyfile(ROOT / "engine/autoarena.py", client)
        executable = root / "bin/autoarena"
        executable.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " "
                              + shlex.quote(str(client)) + ' "$@"\n')
        executable.chmod(0o755)
    logs.mkdir()
    env = method_variables(definition, setup=True)
    values = {**runtime.values(definition, run.path), "uv": shutil.which("uv") or "uv"}
    for index, command in enumerate(definition["method"]["environment"]["setup"], 1):
        _run(runtime.command(command, values), root / "source", env,
             logs / f"{index:02d}-method.log", run.path / "engine/stop.json")
    _check(run, definition, logs / "readiness.log")
    runtime.write_once(receipt, {"plan": plan, "ready": True,
                                 "client_module_sha256": hashlib.sha256(client.read_bytes()).hexdigest()})


def _check(run, definition, log):
    from engine.runtime import lifecycle as runtime
    root = Path(definition["environment"]["root"])
    env = method_variables(definition, setup=True)
    values = runtime.values(definition, run.path)
    _run(runtime.command(definition["method"]["environment"]["check"], values),
         root / "source", env, log, run.path / "engine/stop.json")
    directory = runtime.method_directory(definition)
    for operation in ("start", "resume"):
        argv = runtime.command(definition["method"][operation], values)
        target = argv[0]
        if "/" in target and not Path(target).is_absolute():
            target = str(directory / target)
        executable = shutil.which(target, path=env["PATH"])
        if not executable:
            raise EnvironmentError(f"method {operation} entry point is not executable: {argv[0]}")
