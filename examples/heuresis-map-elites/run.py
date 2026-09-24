"""Own native Heuresis lifecycle and adapt its file queue to AutoArena's API."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CREDENTIAL_NAMES = ("CLAUDE_CODE_USE_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION", "AWS_BEARER_TOKEN_BEDROCK")


def native_environment():
    env = os.environ.copy()
    settings = Path.home() / ".claude/settings.json"
    if settings.is_file():
        values = json.loads(settings.read_text()).get("env", {})
        for key in CREDENTIAL_NAMES:
            if key not in env and key in values:
                env[key] = str(values[key])
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["QD_SEED_OPENCODE_CACHE"] = "0"
    return env


def check():
    import heuresis
    from heuresis import preflight
    from heuresis.tasks.nanogpt import objective
    if not shutil.which(str(Path(sys.prefix) / "bin/heuresis")) or not shutil.which("claude"):
        raise RuntimeError("heuresis and claude executables are required")
    if error := preflight.check_bwrap():
        raise RuntimeError(error)
    # No GPU, no research and no credentials in readiness output.
    print("Heuresis native controller, gated adapter, Claude and bwrap ready")


def configure(benchmark, scratch, resume):
    import yaml
    import heuresis.tasks.nanogpt.adapter as native
    task = benchmark.task
    try:
        from autoarena import objective_spec
    except ModuleNotFoundError:
        from engine.autoarena import objective_spec
    objective = objective_spec(task["objective"])
    ranking = objective["target"]["metric"]
    package = Path(native.__file__).parent
    local = json.loads((ROOT / "settings.json").read_text())
    (package / "baseline_scores.yaml").write_text(yaml.safe_dump({"metric": ranking, "baseline": benchmark.reference_result.get("metrics", {}).get(ranking), "objective": "min"}))
    statement = ("Optimize this task's declared target. Quality is an eligibility threshold only. "
                 "All benchmark constraints and instruments remain fixed. Start from the supplied seed code.\n\n"
                 + json.dumps(task, indent=2) + "\n\nThis run's measured reference:\n"
                 + json.dumps(benchmark.reference_result, indent=2))
    (package / "problem.j2").write_text(statement)
    (package / "description.md").write_text(f"AutoArena {task['task_id']}: minimize {ranking} behind the registered gates.\n")
    template = (ROOT / "support/nanogpt/prompts/cell_ideator_prompt.j2").read_text()
    (package / "prompts/cell_ideator_prompt.j2").write_text(
        "{% set metric_name = " + json.dumps(ranking) + " %}\n\n" + template)
    schema = (package / "idea_schema.md").read_text()
    (package / "idea_schema.md").write_text(schema.replace("val_bpb", ranking))
    task_config = yaml.safe_load((package / "task_config.yaml").read_text())
    task_config.update(description=f"Minimize {ranking} subject to the frozen AutoArena task", requirements=None)
    task_config["verify"] = {
        "stdout": "run.log",
        "min_stdout_bytes": 10000,
        "evidence_description": (
            "The benchmark executes train.py and returns canonical training stdout "
            "with progress entries and its immutable METRICS_JSON line. "
            "Dispatch writes that returned evidence to run.log. If evidence is "
            "suspicious, verification retrieves the original benchmark stdout "
            "for regrading; it does not purchase another training run."
        ),
        "invariants": json.dumps(task, indent=2),
    }
    # Supply the task's target and reference to novelty review.
    task_config.setdefault("novelty_anchor", {})["search_context"] = {"baseline": "this run's frozen reference", "objective": ranking}
    (package / "task_config.yaml").write_text(yaml.safe_dump(task_config))
    native_state = scratch / "native-state.json"
    settings = {key: local[key] for key in ("agent", "model", "num_ideators", "executor_timeout", "ideator_timeout", "reviewer_timeout", "enable_judge", "count_valid", "memory")}
    settings.update(experiment_name=benchmark.run_dir.name, num_iterations=local["rounds"], gpus=[], max_parents=5, session_reset_every=10, novelty_threshold=2, novelty_max_rounds=3)
    if resume:
        settings["resume_exp_id"] = json.loads(native_state.read_text())["experiment_id"]
    config = {"task": "nanogpt", "strategy": "map_elites", "settings": settings, "strategy_config": {key: local[key] for key in ("cell_empty_weight", "cell_crossover_rate", "go_explore_alpha")}}
    config_file = scratch / "launch.yaml"
    config_file.write_text(yaml.safe_dump(config))
    env = native_environment()
    native_tmp = Path("/tmp") / ("ara-" + hashlib.sha256(str(benchmark.run_dir).encode()).hexdigest()[:12])
    # Upstream places its hashed Unix socket under tempfile.gettempdir(). The
    # environment root can be too long for AF_UNIX even with a short filename.
    env.update(TMPDIR=str(native_tmp), TMP=str(native_tmp), TEMP=str(native_tmp))
    env.update(AUTOARENA_SEED_DIR=str(benchmark.task_dir / "code"), AUTOARENA_RUNS_ROOT=str(scratch / "native"), AUTOARENA_QUEUE_DIR=str(scratch / "queue"), AUTOARENA_DISPATCH_SH=str(ROOT / "dispatch.py"), AUTOARENA_NATIVE_VENV=sys.prefix, AUTOARENA_NATIVE_STATE=str(native_state), AUTOARENA_NATIVE_HISTORY=str(scratch / "measured-history.jsonl"))
    return config_file, env


def atomic_json(path, value):
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps(value, indent=2) + "\n")
    staging.replace(path)


def pause(scratch, token, kind, error):
    atomic_json(scratch / "pause-required.json", {"request_id": token, "kind": kind, "error": str(error)})


def publish_result(scratch, request, done, result, idea, parent):
    atomic_json(done / "arena_result.json", result)
    stdout = Path(result["logs"]["stdout"]).read_text(errors="replace")
    stderr = Path(result["logs"]["stderr"]).read_text(errors="replace")
    if result["status"] != "ok" and "METRICS_JSON:" not in stdout:
        stdout += "\n--- canonical stderr ---\n" + stderr
    (done / "run.log").write_text(stdout)
    (done / "run.err").write_text(stderr)
    history = scratch / "measured-history.jsonl"
    seen = {json.loads(line)["request_id"] for line in history.read_text().splitlines()} if history.exists() else set()
    if request.name not in seen:
        with history.open("a") as stream:
            stream.write(json.dumps({"request_id": request.name, "idea": idea, "parent_id": parent, "result": result}) + "\n")
    atomic_json(done / "complete.json", {"status": result["status"]})
    marker = scratch / "pause-required.json"
    if marker.exists():
        state = json.loads(marker.read_text())
        if state.get("request_id") == request.name and state.get("kind") == "delivery_unresolved":
            marker.unlink()


def handle(benchmark, scratch, request):
    """Submit once, then reconcile only that saved request until delivery settles."""
    token = request.name
    done = scratch / "queue/done" / token
    if (done / "complete.json").exists():
        return
    done.mkdir(parents=True, exist_ok=True)
    attempted = done / "submission-attempted.json"
    try:
        research = json.loads((request / "research_log.json").read_text())
        if (request / "prepare.py").read_bytes() != (benchmark.task_dir / "code/prepare.py").read_bytes():
            raise ValueError("Immutable prepare.py was changed")
        candidate = benchmark.workspace / "candidates" / token
        if not candidate.exists():
            shutil.copytree(benchmark.task_dir / "code", candidate)
            shutil.copyfile(request / "train.py", candidate / "train.py")
        prompt = (request / ".prompt.txt").read_text()
        idea = prompt.split("<!--IDEA-BEGIN-->", 1)[1].split("<!--IDEA-END-->", 1)[0].strip()
        parent_file = request / "arena_parent.json"
        parent = json.loads(parent_file.read_text())["candidate_id"] if parent_file.exists() else "reference"
    except Exception as error:
        # A rejection, returned to the executor as an unsuccessful attempt. It does not
        # pause the lane: the immutable instrument and the request's own shape are
        # enforced again by the engine at measurement time, and a refusal the agent can
        # read and react to is the whole of what this side owes it.
        atomic_json(done / "complete.json", {"error": str(error), "kind": "request_rejected"})
        return
    payload = {"request_id": token, "research_log": research, "parent_id": parent, "idea": idea}
    if not attempted.exists():
        # Persist BEFORE submission. If the process dies around the call, an
        # absent response is never permission to purchase another measurement.
        atomic_json(attempted, payload)
        try:
            result = benchmark.evaluate(source=candidate, request_id=token, research_log=research, idea={"text": idea}, parent_id=parent)
        except Exception as error:
            pause(scratch, token, "delivery_unresolved", error)
        else:
            try:
                publish_result(scratch, request, done, result, idea, parent)
            except Exception as error:
                pause(scratch, token, "delivery_unresolved", error)
            return
    # Poll the existing identity. This is read-only and cannot launch GPU work.
    try:
        observed = benchmark.experiment_status(token)
        atomic_json(done / "delivery-status.json", observed)
        status = observed.get("status")
        if status == "completed":
            results = observed.get("results")
            if not isinstance(results, list) or len(results) != 1:
                raise RuntimeError("Canonical completed request did not return exactly one result")
            publish_result(scratch, request, done, results[0], idea, parent)
        elif status == "failed" and (observed.get("error") or {}).get("type") not in {"WorkerLost", "OwnershipUnknown"}:
            message = (observed.get("error") or {}).get("message", "Canonical request failed")
            atomic_json(done / "complete.json", {"error": message, "kind": "request_rejected"})
            marker = scratch / "pause-required.json"
            if marker.exists():
                state = json.loads(marker.read_text())
                if state.get("request_id") == token and state.get("kind") == "delivery_unresolved":
                    marker.unlink()
        else:
            pause(scratch, token, "delivery_unresolved", f"Canonical request remains {status}; inspect original ownership, never resubmit")
    except Exception as error:
        pause(scratch, token, "delivery_unresolved", error)


def resume_preflight(benchmark, scratch):
    """Settle every queued benchmark delivery before the native controller resumes."""
    if not (scratch / "native-state.json").exists():
        artifacts = list((scratch / "native").rglob("*"))
        pending = list((scratch / "queue/pending").iterdir())
        if artifacts or pending or list(scratch.glob("native*.log")):
            raise RuntimeError("Native state identity is missing despite existing work; resume is blocked")
        # Controller failed before creating its native search (for example while
        # rendering the task). There is no iteration or model session to restart.
        return False
    for request in sorted((scratch / "queue/pending").iterdir()):
        if request.name.startswith(".") or not request.is_dir():
            continue
        done = scratch / "queue/done" / request.name
        if not (done / "submission-attempted.json").exists():
            raise RuntimeError("Saved request has no proven submission; cannot resume its interrupted native executor")
        if not (done / "complete.json").exists():
            handle(benchmark, scratch, request)
        completion = json.loads((done / "complete.json").read_text()) if (done / "complete.json").exists() else {}
        rejected = completion.get("kind") == "request_rejected"
        if not rejected and not (done / "arena_result.json").exists():
            raise RuntimeError("Original request remains unresolved or rejected; native resume is blocked")
        # Nothing further is required of the method's own records. Whether its sqlite
        # store, its iteration log and its workspace agree with each other is Heuresis's
        # business; none of them is read by the engine, and refusing a resume over them
        # strands the lane's remaining budget without protecting a recorded number.
    if (scratch / "pause-required.json").exists():
        raise RuntimeError("A recorded native/API pause remains; native resume is blocked")
    return True


def snapshot(scratch, destination):
    # Persist native scientific artifacts without environments, session credentials,
    # binaries or scratch-only sockets. Never copy a sibling search.
    excluded = {".venv", ".venv_extra", ".claude", ".agent-bin", ".local", ".cache", ".config", ".bin", "__pycache__"}
    for source in scratch.rglob("*"):
        relative = source.relative_to(scratch)
        if any(part in excluded for part in relative.parts) or source.is_symlink() or not source.is_file():
            continue
        if source.name.endswith((".sock", "-shm", "-wal")):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    from autoarena import Benchmark
    benchmark = Benchmark.from_context()
    identity = hashlib.sha256(str(benchmark.run_dir).encode()).hexdigest()[:16]
    scratch = benchmark.workspace / "scratch" / identity
    scratch.mkdir(parents=True, exist_ok=True)
    for path in ("queue/pending", "queue/done", "native"):
        (scratch / path).mkdir(parents=True, exist_ok=True)
    native_resume = resume_preflight(benchmark, scratch) if args.resume else False
    config, env = configure(benchmark, scratch, native_resume)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    (benchmark.workspace / "native-scratch.json").write_text(json.dumps({"scratch": str(scratch)}) + "\n")
    log = scratch / (f"native-resume-{time.time_ns()}.log" if args.resume else "native.log")
    try:
        with log.open("a") as stream:
            process = subprocess.Popen([str(Path(sys.prefix) / "bin/heuresis"), "--launch-config", str(config), "--no-memory"], env=env, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                for request in sorted((scratch / "queue/pending").iterdir()):
                    if request.is_dir() and not request.name.startswith("."):
                        handle(benchmark, scratch, request)
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f"Native Heuresis exited {process.returncode}; see {log}")
        if (scratch / "pause-required.json").exists():
            raise RuntimeError("Native/API incompatibility recorded; review pause-required.json")
        target = json.loads((ROOT / "settings.json").read_text())["rounds"]
        (scratch / "validation.json").write_text(json.dumps({"rounds": target, "benchmark_budget_unchanged": True}, indent=2) + "\n")
        benchmark.finish(f"Native Heuresis controller exited after its own search; {target} rounds configured")
    finally:
        snapshot(scratch, benchmark.workspace / "native-record")


if __name__ == "__main__":
    main()
