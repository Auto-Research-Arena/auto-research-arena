"""Start one method controller; research and native state belong to that method.

Linux foreground supervisor. Evaluations use the existing durable workers and
canonical benchmark accounting. No automatic research retries or round loop.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

from engine import ROOT
from engine.evaluation.task import accelerator_error, load_task
from engine.compute import ComputeError, load_backend, resolve_gpu_ids
from engine.evaluation import reference, protocol
from engine.records import collect, runs
from engine.runtime import dispatch, inputs, source, environment


class EngineError(Exception):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(inputs.read_captured_file(path))


def write_once(path, value):
    dispatch._create_file(Path(path), encoded(value))


def path(value):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise EngineError("paths must be explicit absolute locations")
    return Path(value).resolve()


@contextmanager
def lock(location, *, nonblocking=False):
    location = path(str(location))
    with location.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as error:
            raise EngineError("this run already has an owner") from error
        yield handle


def interface(document, *, template=False):
    """Minimal current launch contract; templates may omit installation recipes."""
    if not isinstance(document, dict):
        raise EngineError("interface must be a JSON object")
    document = {"version": 3, **document}
    if type(document["version"]) is not int or document["version"] != 3:
        raise EngineError("unsupported interface version")
    required = {"version", "id", "entry_type", "start", "resume"}
    if not template:
        required.add("environment")
    allowed = required | {"environment", "workers", "prompt_file"}
    missing = required - document.keys()
    if missing:
        raise EngineError("interface missing required fields: " + ", ".join(sorted(missing)))
    if set(document) - allowed:
        raise EngineError("unknown interface fields: " + ", ".join(sorted(set(document) - allowed)))
    if document["entry_type"] not in ("cli", "prompt"):
        raise EngineError("entry_type must be cli or prompt")
    prompt = document.get("prompt_file")
    if document["entry_type"] == "prompt":
        if (not isinstance(prompt, str) or not prompt.strip() or "\0" in prompt or "\\" in prompt
                or Path(prompt).is_absolute() or ".." in Path(prompt).parts or not Path(prompt).name):
            raise EngineError("prompt entry_type requires prompt_file as a relative path inside the method source")
    elif "prompt_file" in document:
        raise EngineError("prompt_file is only for prompt entry_type")
    if not isinstance(document["id"], str) or not protocol.CANDIDATE_ID.fullmatch(document["id"]):
        raise EngineError("invalid method id")
    if type(document.get("workers", 1)) is not int or document.get("workers", 1) < 1:
        raise EngineError("workers must be a positive integer")
    for key in ("start", "resume"):
        argv = document[key]
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x or "\0" in x for x in argv):
            raise EngineError(f"{key} must be an argv array, not a shell string")
    if "environment" in document:
        try:
            environment.validate(document["environment"])
        except environment.EnvironmentError as error:
            raise EngineError(str(error)) from error
    return document


def _launch_prompt(method, source_root):
    if method["entry_type"] == "cli":
        return None
    root = source_root.resolve()
    location = root / method["prompt_file"]
    if not location.resolve().is_relative_to(root):
        raise EngineError("prompt_file must stay inside the method source")
    try:
        prompt = inputs.read_captured_file(location).decode("utf-8")
    except (OSError, UnicodeError, ValueError) as error:
        raise EngineError("prompt_file must be a readable UTF-8 file") from error
    if not prompt.strip():
        raise EngineError("prompt_file must not be empty")
    return prompt


def values(definition, run_dir):
    bindings = definition["config"].get("bindings", {})
    fixed = {"python": sys.executable, "source": str(method_directory(definition)),
             "run_dir": str(run_dir), "workspace": str(run_dir / "workspace"),
             "context": str(run_dir / "engine/context.json"),
             "session_id": definition["session_id"], "task_dir": str(run_dir / "engine/task")}
    if set(bindings) & set(fixed):
        raise EngineError("bindings may not override engine-owned paths or session identity")
    return {**bindings, **fixed}


def command(template, bindings):
    def replace(match):
        key = match.group(1)
        if key not in bindings:
            raise EngineError(f"missing run binding: {key}")
        return bindings[key]
    return [re.sub(r"\{([a-z_][a-z_0-9]*)\}", replace, part) for part in template]


def validate_commands(method, bindings):
    """Resolve every declared command before setup or reference measurement."""
    recipe = method["environment"]
    for template in [method["start"], method["resume"], recipe["check"]]:
        command(template, bindings)
    for template in recipe["setup"]:
        command(template, {**bindings, "uv": shutil.which("uv") or "uv"})


def method_directory(definition):
    return Path(definition["environment"]["root"]) / "source"


def open_run(run_dir):
    run_dir = path(str(run_dir))
    run = runs.open_run(run_dir)
    definition = read(run_dir / "engine/definition.json")
    if hashlib.sha256(encoded(definition)).hexdigest() != run.meta.get("engine_definition_sha256"):
        raise EngineError("frozen engine definition changed")
    if definition.get("version") != 1 or definition.get("api_version") != 1 or definition.get("context_version") != 5:
        raise EngineError("unsupported run format; this release requires the current API context")
    method = interface(definition["method"])
    expected = {"method_id": method["id"], "engine_interface": method}
    if (definition["config"]["run_dir"] != str(run_dir) or run.method_id != method["id"]
            or run.meta.get("method_snapshot") != expected
            or run.meta.get("method_snapshot_sha256") != runs._json_sha256(expected)
            or run.meta.get("task_snapshot") != run.task().raw
            or run.meta.get("task_snapshot_sha256") != runs._json_sha256(run.task().raw)
            or run.task_id != definition["config"]["task"]
            or run.meta.get("lanes") != method.get("workers", 1)):
        raise EngineError("run identity, task or method differs from the frozen definition")
    config = definition["config"]
    package = run_dir / "engine/task"
    if read(package / "task.json") != run.recorded_task():
        raise EngineError("frozen task package definition changed")
    for name, digest in definition["task_files"].items():
        if hashlib.sha256(inputs.read_captured_file(path(str(package / "code" / name)))).hexdigest() != digest:
            raise EngineError("frozen task package code changed")
    compute = {"backend": config["compute"]["backend"],
               "config": {k: v for k, v in config["compute"].items() if k != "backend"}}
    if any(run.meta.get("compute", {}).get(k) != v for k, v in compute.items()):
        raise EngineError("recorded compute binding changed")
    return run, definition


def binding(run_dir):
    run, _ = open_run(run_dir)
    refs = [v for v in run.views() if v.is_reference_launch]
    if len(refs) != 1 or not refs[0].resolved or refs[0].status != "ok" or reference._reference_mismatches(run.task(), refs[0].metrics):
        raise EngineError("reference is not verified; preserve existing work, never remeasure")
    return {name: hashlib.sha256(inputs.read_captured_file(run.path / name)).hexdigest()
            for name in ("run.json", "engine/definition.json")}


def load_config(config_path):
    """Resolve launch paths relative to this repository before freezing them."""
    from engine.compute.storage import cache_environment
    config = read(Path(config_path))
    required = {"task", "method", "run_dir", "compute"}
    if not isinstance(config, dict) or not required <= config.keys() or set(config) - required - {"source", "bindings", "scratch_root", "research_llm"}:
        raise EngineError("run config needs task, method, run_dir and compute; source, bindings, scratch_root and research_llm are optional")
    if not isinstance(config.get("research_llm", {}), dict):
        raise EngineError("research_llm must be an object of model metadata and an optional check command")
    if not isinstance(config.get("bindings", {}), dict) or any(
            not isinstance(k, str) or not isinstance(v, str) or not v or "\0" in v
            for k, v in config.get("bindings", {}).items()):
        raise EngineError("bindings must contain nonempty strings")
    if not isinstance(config["compute"], dict):
        raise EngineError("compute must be an object")
    env = config["compute"].get("env")
    if env is not None and (not isinstance(env, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in env.items())):
        raise EngineError("compute.env must map string names to string values")
    for key in ("method", "source", "run_dir", "scratch_root"):
        if key in config:
            location = Path(config[key]).expanduser()
            config[key] = str((location if location.is_absolute() else ROOT / location).resolve())
    config.setdefault("source", str(Path(config["method"]).parent))
    compute = config.get("compute") or {}
    if compute.get("backend") == "local":
        defaults = cache_environment()
        defaults.update(UV_PROJECT_ENVIRONMENT=str(ROOT / ".local/measurement"), UV_NO_SYNC="1")
        compute["env"] = {**defaults, **compute.get("env", {})}
        compute.setdefault("mounts", {"~/.cache/autoresearch": str(ROOT / "data/autoresearch")})
        for key, value in compute["env"].items():
            if key in defaults and key not in ("UV_LINK_MODE", "UV_NO_SYNC"):
                location = Path(value).expanduser()
                compute["env"][key] = str(location if location.is_absolute() else ROOT / location)
        compute["mounts"] = {alias: str(Path(value) if Path(value).is_absolute() else ROOT / value)
                             for alias, value in compute["mounts"].items()}
    return config


def launch_paths(config, session_id):
    """Check the existing launch path rules without creating run or scratch state."""
    root = path(config["run_dir"])
    if root.exists():
        raise EngineError("run directory already exists; use resume, never start over")
    if not runs.RUN_ID_PATTERN.fullmatch(root.name):
        raise EngineError("invalid run directory name")
    method_source = path(config["source"])
    if root.is_relative_to(method_source) or method_source.is_relative_to(root):
        raise EngineError("method source and run state must be separate trees")
    scratch = path(config.get("scratch_root", str(ROOT / ".local")))
    if scratch == Path("/"):
        raise EngineError("scratch_root must not be the filesystem root")
    environment_root = (scratch / "environments" / session_id).resolve()
    if any(environment_root.is_relative_to(tree) or tree.is_relative_to(environment_root)
           for tree in (method_source, root)):
        raise EngineError("private environment must be separate from method source and run records")
    return root, method_source, environment_root


def check_compute(backend, task, workers):
    """Check the task hardware against the devices this run will actually use."""
    problems = backend.preflight()
    if problems:
        raise EngineError("compute preflight failed: " + "; ".join(problems))
    devices = resolve_gpu_ids(workers * task.launch["gpus_per_launch"], visible=backend.visible_devices)
    problem = accelerator_error(task, devices, backend.gpu_names(devices))
    if problem:
        raise EngineError(problem)
    return devices


def prepare(config_path):
    config = load_config(config_path)
    session_id = str(uuid.uuid4())
    root, method_source, environment_root = launch_paths(config, session_id)
    config["run_dir"] = str(root)
    method = interface(read(path(config["method"])))
    task = load_task(config["task"])
    definition = {"version": 1, "api_version": 1, "context_version": 5, "method": method, "config": config,
                  "session_id": session_id,
                  "source_lock": source.lock(method_source)}
    definition["launch_prompt"] = _launch_prompt(method, method_source)
    definition["environment"] = {"root": str(environment_root),
                                 "client_sha256": environment.client_digest()}
    definition["runner_environment"] = {key: os.environ.get(key) for key in environment.RUNNER_ENV_KEYS}
    task_files = {str(p.relative_to(task.source_root)): inputs.read_captured_file(p)
                  for p in sorted(task.source_root.rglob("*"))
                  if p.is_file() and not any(part in {".git", "__pycache__", ".venv"}
                                            for part in p.relative_to(task.source_root).parts)}
    definition["task_files"] = {name: hashlib.sha256(data).hexdigest() for name, data in task_files.items()}
    substitutions = values(definition, root)
    validate_commands(method, substitutions)
    backend = load_backend(config["compute"])
    check_compute(backend, task, method.get("workers", 1))
    root.mkdir(parents=True, exist_ok=False)
    (root / "engine").mkdir()
    (root / "workspace").mkdir()
    package = root / "engine/task"
    package.mkdir()
    write_once(package / "task.json", task.raw)
    for name, data in task_files.items():
        destination = package / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        dispatch._create_file(destination, data)
    snapshot = {"method_id": method["id"], "engine_interface": method}
    run = runs.create(root.parent, task=task, method_id=method["id"], run_id=root.name,
                      lanes=method.get("workers", 1), research_model=(config.get("research_llm") or {}).get("model") or config.get("bindings", {}).get("model"),
                      compute={"backend": config["compute"]["backend"],
                               "config": {k: v for k, v in config["compute"].items() if k != "backend"},
                               "describe": backend.describe()}, method_snapshot=snapshot,
                      extra={"engine_definition_sha256": hashlib.sha256(encoded(definition)).hexdigest()})
    write_once(root / "engine/definition.json", definition)
    return run


def _reference(run, definition, *, retry_setup=False):
    try:
        environment.prepare(run, definition, retry_setup=retry_setup)
    except (environment.EnvironmentError, source.SourceError, OSError, subprocess.SubprocessError) as error:
        _state(run, {"status": "failed", "phase": "environment", "error": str(error)})
        raise EngineError(str(error)) from error
    refs = [v for v in run.views() if v.is_reference_launch]
    if refs:
        binding(run.path)
        return
    if (run.path / "engine/reference-started.json").exists() or run.views():
        raise EngineError("interrupted reference initialization; no automatic replay")
    backend = None
    task = run.task()
    try:
        try:
            backend = load_backend(definition["config"]["compute"])
            check_compute(backend, task, run.meta["lanes"])
            workers = backend.workers(lanes=run.meta["lanes"], gpus_per_lane=task.launch["gpus_per_launch"])
            if len(workers) != run.meta["lanes"]:
                raise EngineError("compute did not supply the declared worker capacity")
            devices = [device for worker in workers for device in worker.gpu_ids]
            problem = accelerator_error(task, devices, backend.gpu_names(devices))
            if problem:
                raise EngineError(problem)
        except (ComputeError, EngineError, OSError) as error:
            _state(run, {"status": "failed", "phase": "compute", "error": str(error)})
            raise EngineError(str(error)) from error
        write_once(run.path / "engine/reference-started.json", {"at": runs.utc_now()})
        original = run.path / "engine/task/code"
        reference_source = run.path / "candidates/reference"
        shutil.copytree(original, reference_source, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))
        (run.path / "engine/source-identities").mkdir()
        identity = program_identity(task, reference_source)
        archive_source(run.path, "reference", reference_source, identity)
        write_once(run.path / "engine/source-identities/reference.json", identity)
        reference._reference_launch(run, task, workers[0], reference_source, definition["config"]["compute"]["backend"])
    finally:
        if backend is not None:
            backend.release()
    binding(run.path)


def context(run, definition):
    if definition.get("context_version") != 5 or definition.get("api_version") != 1:
        raise EngineError("unsupported API context version")
    refs = [v for v in run.views() if v.is_reference_launch and v.resolved]
    return {"version": 5, "api_version": 1, "run_dir": str(run.path),
            "workspace": str(run.path / "workspace"), "source": values(definition, run.path)["source"],
            "task": run.recorded_task(), "task_dir": str(run.path / "engine/task"),
            "method": definition["method"], "bindings": definition["config"].get("bindings", {}),
            "session_id": definition["session_id"], "runner_root": str(ROOT),
            "reference_result": {"candidate_id": refs[0].candidate_id, "launch_uuid": refs[0].launch_uuid,
                                 "status": refs[0].status, "metrics": refs[0].metrics} if refs else None,
            "evaluate": [sys.executable, "-m", "engine", "evaluate", "--run", str(run.path)],
            "finish": [sys.executable, "-m", "engine", "finish", "--run", str(run.path)],
            "status": [sys.executable, "-m", "engine", "status", "--run", str(run.path)],
            "experiment_status": [sys.executable, "-m", "engine", "experiment-status", "--run", str(run.path)]}


def _entry_input(ctx, launch_prompt=None):
    if ctx["method"]["entry_type"] == "cli":
        return ""
    if not isinstance(launch_prompt, str) or not launch_prompt.strip():
        raise EngineError("frozen launch prompt is missing")
    return launch_prompt


def _state(run, value):
    location = run.path / "engine/state.json"
    temporary = location.with_name(".state-" + uuid.uuid4().hex)
    write_once(temporary, {"at": runs.utc_now(), **value})
    os.replace(temporary, location)


def supervise(run_dir, *, resume=False, retry_setup=False):
    from engine.runtime.service import serve
    if retry_setup and not resume:
        raise EngineError("setup retry requires explicit resume")
    run, definition = open_run(run_dir)
    with lock(run.path / "engine/controller.lock", nonblocking=True) as lease:
        with lock(run.path / "engine/control.lock"):
            if (run.path / "engine/finished.json").exists():
                raise EngineError("method already declared completion")
            if (run.path / "engine/stop.json").exists():
                raise EngineError("run was explicitly stopped; no implicit restart")
        if resume:
            prior = read(run.path / "engine/state.json")
            if prior.get("status") not in {"paused", "failed"} or active_evaluations(run) or any(not v.resolved for v in run.views()):
                raise EngineError("resume needs a settled prior controller exit; interrupted ownership requires recovery")
        attempted = run.path / "engine/controller-started.json"
        if not resume and attempted.exists():
            raise EngineError("controller was already started; use explicit resume")
        key = "resume" if resume and attempted.exists() else "start"
        template = definition["method"][key]
        _state(run, {"status": "preparing"})
        _reference(run, definition, retry_setup=retry_setup)
        ctx = context(run, definition)
        context_path = run.path / "engine/context.json"
        if not context_path.exists():
            write_once(context_path, ctx)
        elif read(context_path) != ctx:
            raise EngineError("generated context differs from frozen inputs")
        if not attempted.exists():
            write_once(attempted, {"at": runs.utc_now()})
        argv = command(template, values(definition, run.path))
        attempt = run.path / "engine" / ("attempt-" + uuid.uuid4().hex)
        attempt.mkdir()
        (attempt / "input.txt").write_bytes(_entry_input(ctx, definition.get("launch_prompt")).encode("utf-8"))
        from engine.runtime.environment import method_variables
        environment = {**method_variables(definition), "AUTOARENA_CONTEXT": str(context_path)}
        # Context JSON is the only path handoff; do not inherit stale shortcuts
        # from an outer run or the operator's shell.
        for name in ("AUTOARENA_TASK_DIR", "AUTOARENA_WORKSPACE", "AUTOARENA_RUN_DIR"):
            environment.pop(name, None)
        previous = {}
        process = None
        requested = False
        def interrupted(_number, _frame):
            nonlocal requested
            requested = True
        try:
            for number in (signal.SIGINT, signal.SIGTERM):
                previous[number] = signal.signal(number, interrupted)
            with serve(run.path), (attempt / "input.txt").open("rb") as incoming, (attempt / "stdout.log").open("xb") as out, (attempt / "stderr.log").open("xb") as err:
                with lock(run.path / "engine/control.lock"):
                    if requested or (run.path / "engine/stop.json").exists():
                        _state(run, {"status": "stopped"})
                        return 130
                    process = subprocess.Popen(argv, cwd=method_directory(definition), env=environment,
                                               stdin=incoming, stdout=out, stderr=err,
                                               start_new_session=True, pass_fds=(lease.fileno(),))
                _state(run, {"status": "running", "pid": process.pid, "attempt": str(attempt)})
                completed = False
                drained_at = 0.0
                while process.poll() is None:
                    if requested or (run.path / "engine/stop.json").exists():
                        requested = True
                        stop(run.path, "operator signal or stop request")
                        break
                    # After method completion and settled accepted jobs, terminate
                    # a lingering controller. Poll dispatch state at an interval.
                    now = time.monotonic()
                    if (run.path / "engine/finished.json").exists() and now - drained_at > 2.0:
                        drained_at = now
                        if not active_evaluations(run):
                            completed = True
                            break
                    time.sleep(0.1)
                if requested or (completed and process.poll() is None):
                    _terminate(process)
                code = process.wait()
                # `completed` means the method declared completion and its evaluations
                # drained, so the run finished even though terminating the lingering
                # controller gives a signal exit code. Reporting that as `failed` would
                # invert the outcome of a successful run.
                state = ("stopped" if requested
                         else "finished" if completed or ((run.path / "engine/finished.json").exists() and code == 0)
                         else "paused" if code == 0 else "failed")
                record = {"status": state, "exit_code": code, "attempt": str(attempt)}
                if completed and code != 0:
                    record["controller_outlived_completion"] = True
                _state(run, record)
                if requested:
                    return 130
                return 0 if completed else code
        finally:
            if process is not None and process.poll() is None:
                _terminate(process)
            for number, handler in previous.items():
                signal.signal(number, handler)


def _terminate(process):
    # Only the group created by this Popen. Native nested sessions require their
    # controller's signal handler (the bundled stream transport provides one).
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def active_evaluations(run):
    records = [dispatch.status(run.path, p.parent.name) for p in (run.path / "method/dispatch").glob("*/job.json")]
    return [row for row in records if row["status"] not in {"completed", "failed"}]


def stop(run_dir, reason="user requested stop"):
    run, _ = open_run(run_dir)
    with lock(run.path / "engine/control.lock"):
        target = run.path / "engine/stop.json"
        if not target.exists():
            write_once(target, {"at": runs.utc_now(), "reason": reason})
    return status(run.path)


def finish(run_dir, reason):
    if not isinstance(reason, str) or not reason.strip():
        raise EngineError("an explicit native stopping reason is required")
    binding(run_dir)
    run, _ = open_run(run_dir)
    with lock(run.path / "engine/control.lock"):
        if (run.path / "engine/stop.json").exists():
            raise EngineError("operator stop is not native completion")
        if active_evaluations(run) or any(not v.resolved for v in run.views()):
            raise EngineError("evaluations are not settled; native completion refused")
        write_once(run.path / "engine/finished.json", {"at": runs.utc_now(), "reason": reason})
    return {"method_declared_complete": True, "native_completion_verified": False}


def status(run_dir):
    run, _ = open_run(run_dir)
    state = read(run.path / "engine/state.json") if (run.path / "engine/state.json").exists() else {"status": "prepared"}
    try:
        with lock(run.path / "engine/controller.lock", nonblocking=True):
            owner = False
    except EngineError:
        owner = True
    if not owner and state["status"] in {"running", "preparing"}:
        state = {**state, "status": "interrupted"}
    views = run.views()
    return {**state, "controller_owned": owner, "stop_requested": (run.path / "engine/stop.json").exists(),
            "method_declared_complete": (run.path / "engine/finished.json").exists(),
            "launches": len(views), "resolved": sum(v.resolved for v in views),
            "charged": sum(v.charge().charged for v in views),
            "budget_remaining": max(0, run.task().max_launches - sum(v.charge().charged for v in views)),
            "evaluations_in_flight": len(active_evaluations(run))}


def submit(run_dir, request_id, candidates):
    run, _ = open_run(run_dir)
    with lock(run.path / "engine/control.lock"):
        # Existing request recovery remains available after stop/completion.
        existing = dispatch.lookup(run.path, request_id)
        if existing is None and any((run.path / ("engine/" + name)).exists() for name in ("stop.json", "finished.json")):
            raise EngineError("run is stopped or complete; no new evaluation")
        return dispatch.submit(run.path, candidates, request_id=request_id)


def evaluate_batch(run_dir, candidates):
    binding(run_dir)
    run, definition = open_run(run_dir)
    # Already-submitted work drains even after an operator stop. No new request
    # can be accepted after stop; ambiguous work is never silently retried.
    with lock(run.path / "engine/evaluation.lock"):
        task = run.task()
        candidates = dispatch._candidate_data(candidates)
        views = run.views()
        protocol._check_admission(task, views, [item["candidate_id"] for item in candidates])
        identities = []
        for item in candidates:
            checked = protocol._checked_source(task, item["candidate_id"], Path(item["source"]))
            identities.append(program_identity(task, checked))
        stored = run.path / "engine/source-identities"
        # A resolved, uncharged attempt did not buy the program. Keep its record,
        # but allow a new candidate/request unless another attempt still reserves it.
        blocked = {v.candidate_id for v in views if not v.resolved or v.charge().charged}
        refunded = {v.candidate_id for v in views if v.resolved and not v.charge().charged} - blocked
        seen = {read(p)["sha256"] for p in stored.glob("*.json") if p.stem not in refunded}
        for identity in identities:
            if identity["sha256"] in seen:
                raise EngineError("this task program is already reserved or measured; no new purchase")
            seen.add(identity["sha256"])
        for item, identity in zip(candidates, identities):
            archive_source(run.path, item["candidate_id"], Path(item["source"]), identity)
            write_once(stored / (item["candidate_id"] + ".json"), identity)
        results = protocol.evaluate_batch(run, candidates=candidates, backend_config=definition["config"]["compute"])
        for item, identity in zip(candidates, identities):
            if program_identity(task, Path(item["source"])) != identity:
                raise EngineError("candidate source changed during evaluation; preserve measurements for audit")
        return results


def archive_source(run_dir, candidate_id, original, identity):
    """Keep measured bytes separate from the method's editable workspace."""
    archive = run_dir / "engine/source-snapshots" / candidate_id
    for name, digest in identity["files"].items():
        data = inputs.read_captured_file(original / name)
        if hashlib.sha256(data).hexdigest() != digest:
            raise EngineError("candidate source changed before evaluation")
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)


def program_identity(task, root):
    """Exact declared task-file bytes, not candidate names or guessed semantics.

    Charged or unresolved attempts retain their reservations. A resolved,
    uncharged attempt permits a new request for the same program.
    """
    files = {name: hashlib.sha256(inputs.read_captured_file(path(str(root / name)))).hexdigest()
             for name in sorted(set(task.substrate["mutable"] + task.substrate["immutable"]))}
    return {"files": files, "sha256": hashlib.sha256(encoded({"task": task.raw, "files": files})).hexdigest()}


def evaluate(run_dir, request_id, candidates):
    job = submit(run_dir, request_id, candidates)
    job = dispatch.wait(run_dir, job)
    if job["status"] != "completed":
        raise EngineError("evaluation delivery failed or is unresolved; inspect the original request, never replay")
    return job["results"]


def experiment_status(run_dir, request_id):
    """Inspect only; an absent or interrupted request never launches work."""
    run, _ = open_run(run_dir)
    job = dispatch.lookup(run.path, request_id)
    if job is None:
        return {"request_id": request_id, "status": "not_found"}
    return {"request_id": request_id, **{key: job[key] for key in
            ("status", "results", "error", "job_id") if key in job}}


def summary(run_dir):
    run, _ = open_run(run_dir)
    return collect.collect(run)
