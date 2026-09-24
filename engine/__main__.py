"""CLI lifecycle and evaluation, never a method's research loop."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from engine.runtime import lifecycle as runtime
from engine.runtime import dispatch
from engine.runtime import service
from engine.records import verify


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("run").add_argument("--config", type=Path, required=True)
    readiness = commands.add_parser("check", help="show package and selected-run readiness")
    readiness.add_argument("--config", type=Path)
    readiness.add_argument("--json", action="store_true", help="machine-readable check results")
    commands.add_parser("tasks")
    report = commands.add_parser("submission", help="export a run with its report, source and evidence")
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    for name in ("resume", "status", "stop", "collect", "verify", "finish", "evaluate", "experiment-status"):
        item = commands.add_parser(name)
        location = item.add_mutually_exclusive_group()
        location.add_argument("--run", type=Path)
        location.add_argument("--context", type=Path, help="defaults to AUTOARENA_CONTEXT")
        if name in {"finish", "stop"}:
            item.add_argument("--reason", required=name == "finish", default="user requested stop")
        if name == "resume":
            item.add_argument("--retry-setup", action="store_true",
                              help="retry failed method setup before any measurement starts")
        if name in {"evaluate", "experiment-status"}:
            item.add_argument("--request-id", required=True)
        if name == "evaluate":
            submission = item.add_mutually_exclusive_group(required=True)
            submission.add_argument("--batch", help="JSON batch file, or - for stdin")
            submission.add_argument("--source", type=Path, help="one complete candidate directory")
            item.add_argument("--candidate-id")
            item.add_argument("--parent-id")
            item.add_argument("--idea", help="optional JSON object of native research metadata")
            research = item.add_mutually_exclusive_group()
            research.add_argument("--research-log", help="JSON object with a nonempty ideas list; required for candidates. "
                                  "Status is an optional nonblank string")
            research.add_argument("--research-log-file", type=Path, help="read the research-log JSON object from this file")
    args = parser.parse_args(argv)
    try:
        if getattr(args, "run", None) is not None:
            args.run = runtime.path(str(args.run.absolute()))
        if hasattr(args, "run") and args.run is None:
            context_path = args.context or os.environ.get("AUTOARENA_CONTEXT")
            if not context_path:
                raise runtime.EngineError("supply --run, --context, or AUTOARENA_CONTEXT")
            ctx = runtime.read(Path(context_path))
            if ctx.get("api_version") != 1 or ctx.get("version") != 5:
                raise runtime.EngineError("unsupported API context version")
            args.run = runtime.path(ctx["run_dir"])
        if args.action == "tasks":
            from engine.evaluation.task import available_tasks, load_task
            output = [{"id": name, "title": load_task(name).title} for name in available_tasks()]
        elif args.action == "check":
            from engine.runtime.checks import check
            from engine.runtime.readiness import check_config, render
            output = check_config(args.config) if args.config else check()
            print(json.dumps(output, indent=2) if args.json else render(output))
            return 0 if output["passed"] else 1
        elif args.action == "submission":
            from engine.submission.export import export
            record = export(args.run, args.output)
            output = {"report": str(args.output / "submission.html"), "verdict": record["verdict"]}
        elif args.action == "run":
            run = runtime.prepare(args.config)
            code = runtime.supervise(run.path)
            if code:
                print(f"Controller did not finish successfully; inspect {run.path / 'engine'}", file=sys.stderr)
            return code
        elif args.action == "resume":
            return runtime.supervise(args.run, resume=True, retry_setup=args.retry_setup)
        elif args.action == "evaluate":
            if args.batch is not None:
                if any(value is not None for value in (args.candidate_id, args.parent_id, args.idea, args.research_log, args.research_log_file)):
                    raise runtime.EngineError("candidate-id, parent-id, idea and research-log apply only to --source")
                batch = json.load(sys.stdin) if args.batch == "-" else runtime.read(Path(args.batch))
            else:
                identifier = args.candidate_id or "exp-" + hashlib.sha256(args.request_id.encode()).hexdigest()[:24]
                batch = [{"candidate_id": identifier, "source": str(args.source.absolute()),
                          "parent_id": args.parent_id, "idea": json.loads(args.idea) if args.idea is not None else {}}]
                if args.research_log is not None:
                    batch[0]["research_log"] = json.loads(args.research_log)
                elif args.research_log_file is not None:
                    batch[0]["research_log"] = runtime.read(args.research_log_file)
            if isinstance(batch, list):
                batch = [{**item, "source": str(Path(item["source"]).absolute())}
                         if isinstance(item, dict) and isinstance(item.get("source"), str) else item
                         for item in batch]
            output = service.invoke(args.run, "evaluate", request_id=args.request_id, candidates=batch)
            if args.source is not None:
                output = output[0]
        elif args.action == "experiment-status":
            output = service.invoke(args.run, "experiment-status", request_id=args.request_id)
        elif args.action == "finish":
            output = service.invoke(args.run, "finish", reason=args.reason)
        elif args.action == "stop":
            output = runtime.stop(args.run, args.reason)
        elif args.action == "status":
            output = service.invoke(args.run, "status")
        elif args.action == "collect":
            output = runtime.summary(args.run)
        else:
            run, _ = runtime.open_run(args.run)
            audit = verify.verify(run)
            print(json.dumps(audit.to_dict(), indent=2))
            return 0 if audit.passed else 1
        print(json.dumps(output, indent=2))
        return 0
    except (runtime.EngineError, dispatch.DispatchError, OSError, ValueError, KeyError,
            runtime.protocol.ProtocolError, runtime.runs.RunError, runtime.source.SourceError,
            runtime.inputs.InputError) as error:
        print(f"engine error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
