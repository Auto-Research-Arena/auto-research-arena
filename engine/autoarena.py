"""Standard-library client for the AutoArena experiment API; no research policy."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess


def objective_spec(objective):
    """Copy the explicit objective without changing the task or its scoring."""
    fields = {"comparison_mode", "target", "quality_gate", "tiebreaks", "constraints"}
    if not isinstance(objective, dict) or set(objective) != fields:
        raise ValueError("objective must contain exactly " + ", ".join(sorted(fields)))
    if objective["comparison_mode"] != "gated":
        raise ValueError("objective.comparison_mode must be gated")
    return deepcopy(objective)


class APIError(RuntimeError):
    """A request failed or delivery is unresolved. Inspect its ID before recovery."""


class Benchmark:
    """Connect using the context supplied to a method at launch.

    Evaluation is synchronous. Reusing the same request ID and inputs recovers
    the original response; the client never invents retries or experiment IDs.
    """

    def __init__(self, context):
        if (not isinstance(context, dict)
                or type(context.get("version")) is not int or context["version"] != 5
                or type(context.get("api_version")) is not int or context["api_version"] != 1):
            raise APIError("expected AutoArena context version 5 and API version 1")
        self.context = context
        self.task_dir = Path(context["task_dir"])
        self.workspace = Path(context["workspace"])
        self.run_dir = Path(context["run_dir"])
        self.reference_result = context["reference_result"]

    @classmethod
    def from_context(cls, path=None):
        """Read an explicit context file or AUTOARENA_CONTEXT."""
        location = path or os.environ.get("AUTOARENA_CONTEXT")
        if not location:
            raise APIError("supply a context path or AUTOARENA_CONTEXT")
        try:
            return cls(json.loads(Path(location).read_text()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise APIError(f"invalid API context: {error}") from error

    @property
    def task(self):
        """The frozen task definition. Reading it is the method's choice."""
        return json.loads((self.task_dir / "task.json").read_text())

    def _call(self, operation, args=(), payload=None):
        command = self.context[operation]
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise APIError(f"invalid {operation} command in context")
        if not all(isinstance(x, str) for x in args):
            raise APIError("request IDs, candidate IDs and reasons must be strings")
        try:
            result = subprocess.run(
                command + list(args),
                input=None if payload is None else json.dumps(payload, allow_nan=False),
                text=True, capture_output=True, cwd=self.context["runner_root"],
            )
        except (OSError, ValueError, TypeError) as error:
            raise APIError(f"{operation} could not be delivered: {error}") from error
        if result.returncode:
            raise APIError(result.stderr.strip() or f"{operation} exited {result.returncode}")
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise APIError(f"invalid {operation} response; inspect the original request") from error

    def evaluate(self, source, request_id, *, research_log=None, candidate_id=None, parent_id=None, idea=None):
        """Measure one complete candidate directory; return its raw result object.

        research_log requires a nonempty ideas list; status is optional and must
        be a nonblank string when supplied. Keep the exact research_log on
        retries, including status presence.
        Candidate ID defaults to a stable ID derived from request_id. Optional
        idea/parent_id remain native metadata; the runner never interprets research
        as a search instruction.
        """
        args = ["--request-id", request_id, "--source", str(Path(source).absolute())]
        if candidate_id is not None:
            args += ["--candidate-id", candidate_id]
        if parent_id is not None:
            args += ["--parent-id", parent_id]
        if research_log is not None:
            try:
                args += ["--research-log", json.dumps(research_log, allow_nan=False)]
            except (ValueError, TypeError) as error:
                raise APIError("research_log must be JSON-serializable") from error
        if idea is not None:
            try:
                args += ["--idea", json.dumps(idea, allow_nan=False)]
            except (ValueError, TypeError) as error:
                raise APIError("idea must be JSON-serializable") from error
        return self._call("evaluate", args)

    def evaluate_batch(self, candidates, request_id):
        """Submit candidate_id/source/research_log objects; return results in input order.

        Each research_log follows evaluate's contract, including exact retry payloads.
        """
        return self._call("evaluate", ["--request-id", request_id, "--batch", "-"], candidates)

    def experiment_status(self, request_id):
        """Inspect a request without submitting or re-running it."""
        return self._call("experiment_status", ["--request-id", request_id])

    def status(self):
        """Observe process state, charges and evaluations, not research progress."""
        return self._call("status")

    def finish(self, reason):
        """Declare completion after the method has finished its own feedback."""
        return self._call("finish", ["--reason", reason])


def main(argv=None):
    """Installed CLI: forward experiment operations to the bound runner."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path)
    parser.add_argument("operation", choices=("evaluate", "experiment-status", "status", "finish"))
    args, remaining = parser.parse_known_args(argv)
    try:
        benchmark = Benchmark.from_context(args.context)
        if args.operation == "evaluate":
            for index, argument in enumerate(remaining):
                if argument == "--":
                    break
                option, joined, value = argument.partition("=")
                if option not in {"--source", "--research-log-file", "--batch"}:
                    continue
                if not joined:
                    if index + 1 == len(remaining):
                        continue
                    value = remaining[index + 1]
                    if value.startswith("-") and value != "-":
                        continue
                if option == "--batch" and value == "-":
                    continue
                absolute = str(Path(value).absolute())
                remaining[index if joined else index + 1] = option + "=" + absolute if joined else absolute
        result = benchmark._call(args.operation.replace("-", "_"), remaining)
        print(json.dumps(result, indent=2))
        return 0
    except (APIError, OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
