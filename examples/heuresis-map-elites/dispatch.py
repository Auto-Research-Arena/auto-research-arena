"""In-sandbox file queue client. Saved requests recover the same API experiment."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time

QUEUE = Path("/arena/queue")


def main():
    source = Path("train.py").read_bytes()
    research = json.loads(Path("research_log.json").read_text())
    if not isinstance(research.get("ideas"), list) or not research["ideas"]:
        raise RuntimeError("research_log.json needs a nonempty ideas list")
    if not isinstance(research.get("status"), str) or not research["status"].strip():
        raise RuntimeError("research_log.json needs a nonblank status")
    workspace_id = Path(".workspace_id").read_text().strip()
    identity = hashlib.sha256(source + json.dumps(research, sort_keys=True).encode()).hexdigest()
    token = f"{workspace_id}-{identity[:24]}"
    previous_file = Path(".arena_request_id")
    if previous_file.exists() and previous_file.read_text().strip() != token:
        previous = previous_file.read_text().strip()
        completed = QUEUE / "done" / previous / "complete.json"
        if not completed.is_file():
            raise RuntimeError("Original request feedback is unresolved. Recover the SAME request before changing code or research.")
    pending = QUEUE / "pending" / token
    if not pending.exists():
        staging = QUEUE / "pending" / f".stage-{token}-{os.getpid()}"
        staging.mkdir()
        for name in ("train.py", "prepare.py", "research_log.json", ".prompt.txt", "arena_parent.json"):
            if Path(name).is_file():
                shutil.copyfile(name, staging / name)
        (staging / "identity.json").write_text(json.dumps({"request_id": token, "workspace_id": workspace_id}))
        staging.rename(pending)
    # Persist before waiting; the same files generate the same durable request.
    Path(".arena_request_id").write_text(token + "\n")
    done = QUEUE / "done" / token
    # Allow four hours for setup, training, measurement and feedback delivery.
    deadline = time.monotonic() + 3600 * 4
    print(f"Queued {token}; waiting for canonical evaluation", flush=True)
    while not (done / "complete.json").exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Delivery pending. Recover the SAME request; do not modify or remeasure code.")
        time.sleep(2)
    response = json.loads((done / "complete.json").read_text())
    # The delivered result is written beside the log of the same delivery. Upstream's
    # `python train.py` produced its own log and the grader read the number out of it;
    # here the launch happens remotely, so the log AND its result are carried back
    # together into the workspace. The grading path then reads the result belonging to
    # the log it grades with a plain file read.
    for name in ("run.log", "run.err", "arena_result.json"):
        if (done / name).is_file():
            shutil.copyfile(done / name, name)
    if response.get("error"):
        raise RuntimeError(response["error"])
    print(Path("arena_result.json").read_text(), flush=True)
    if Path("run.err").is_file():
        print("\n".join(Path("run.err").read_text(errors="replace").splitlines()[-40:]), flush=True)


if __name__ == "__main__":
    main()
