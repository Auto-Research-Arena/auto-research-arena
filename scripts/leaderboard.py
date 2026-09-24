#!/usr/bin/env python3
"""Add, check and build the public AutoArena leaderboard."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.submission import leaderboard
from engine.submission.export import write_json


def submission_files(directory):
    paths = sorted(directory.rglob("*.json"))
    for path in paths:
        leaderboard.require(path.name in ("submission.json", "artifacts.json"),
                            f"{path}: put each run in a folder containing submission.json and artifacts.json")
        if path.name == "artifacts.json":
            leaderboard.require(path.with_name("submission.json").is_file(),
                                f"{path}: missing companion submission.json")
    return [path for path in paths if path.name == "submission.json"]


def changed_entries(base_ref, directory):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD"],
        cwd=ROOT, text=True).splitlines()
    # A change to the checks themselves revalidates all accepted exports.
    rules = ("engine/", "tasks/", "scripts/leaderboard.py")
    if any(name.startswith(rules) for name in changed):
        return submission_files(directory)
    return sorted({(ROOT / name).with_name("submission.json") for name in changed
                   if (ROOT / name).resolve().is_relative_to(directory.resolve())
                   and Path(name).name in ("submission.json", "artifacts.json")})


def build(directory, output):
    results = []
    paths = submission_files(directory)
    for path in paths:
        result = leaderboard.check_submission(path)
        relative = path.relative_to(directory)
        result["submission_url"] = "submissions/" + quote(relative.as_posix(), safe="/")
        results.append(result)
    data = leaderboard.build_data(results)
    metadata = leaderboard.read_json((ROOT / "site/methods.json").read_bytes())
    leaderboard.require(isinstance(metadata, dict), "method display metadata must be an object")
    for method_id, info in metadata.items():
        leaderboard.require(isinstance(info, dict) and isinstance(info.get("name"), str)
                            and bool(info["name"].strip()), f"{method_id}: a display name is required")
        for field in ("source_url", "license_url"):
            if field in info:
                leaderboard.public_url(info[field])
        if "license" in info:
            leaderboard.require(isinstance(info["license"], str) and bool(info["license"].strip())
                                and bool(info.get("source_url")) and bool(info.get("license_url")),
                                f"{method_id}: a license needs public source and license links")
    data["method_metadata"] = metadata
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    safe = (text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    template = (ROOT / "site/index.html").read_text()
    (output / "index.html").write_text(template.replace("LEADERBOARD_DATA", safe))
    write_json(output / "leaderboard.json", data)
    write_json(output / "ranking.json", data["overall"])
    for name in ("site.css", "site.js", "submit.html"):
        shutil.copyfile(ROOT / "site" / name, output / name)
    for path in paths:
        target = output / "submissions" / path.relative_to(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        shutil.copyfile(path.with_name("artifacts.json"), target.with_name("artifacts.json"))
    (output / ".nojekyll").touch()
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="convert an exported run to website JSON and optional artifact links")
    add.add_argument("--submission", type=Path, required=True, help="exported local submission.json")
    add.add_argument("--artifacts-url", help="download link to an archive containing both code and logs")
    add.add_argument("--code-url", help="separate code download link, overriding --artifacts-url")
    add.add_argument("--logs-url", help="separate log download link, overriding --artifacts-url")
    add.add_argument("--report-url", help="optional hosted HTML report")
    add.add_argument("--output", type=Path, required=True, help="new site/submissions/<method>/<run>/ directory")
    check = commands.add_parser("check", help="validate committed submission JSONs without network access")
    check.add_argument("entries", nargs="*", type=Path)
    check.add_argument("--directory", type=Path, default=ROOT / "site/submissions")
    check.add_argument("--base-ref", help="check entries changed since this Git ref")
    render = commands.add_parser("build", help="build the static website from accepted entries")
    render.add_argument("--directory", type=Path, default=ROOT / "site/submissions")
    render.add_argument("--output", type=Path, default=ROOT / "runs/leaderboard-site")
    args = parser.parse_args(argv)
    try:
        if args.command == "add":
            leaderboard.require(not args.output.exists(), "output exists; choose a new run directory")
            raw = leaderboard.read_submission(args.submission)
            record = leaderboard.site_record(leaderboard.read_json(raw))
            result = leaderboard.project_site_record(record)
            links = {"code_url": args.code_url or args.artifacts_url,
                     "logs_url": args.logs_url or args.artifacts_url}
            links = {key: value for key, value in links.items() if value}
            if args.report_url:
                links["report_url"] = args.report_url
            links = leaderboard.validate_artifacts(links)
            args.output.mkdir(parents=True)
            write_json(args.output / "submission.json", record)
            write_json(args.output / "artifacts.json", links)
            print(f"Created {args.output}: {result['task_id']}, {len(result['points'])} qualifying measurements")
        elif args.command == "check":
            paths = args.entries or (changed_entries(args.base_ref, args.directory)
                                     if args.base_ref else submission_files(args.directory))
            for path in paths:
                leaderboard.check_submission(path)
                print(f"PASS {path}")
            # Local validation also catches duplicate run identities across PRs.
            leaderboard.build_data([leaderboard.check_submission(path)
                                    for path in submission_files(args.directory)])
            print(f"Checked {len(paths)} submission(s); all local records are consistent.")
        else:
            data = build(args.directory, args.output)
            count = sum(len(task["runs"]) for task in data["tasks"])
            print(f"Built {args.output}: {count} runs across {len(data['tasks'])} tasks")
    except (leaderboard.LeaderboardError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
