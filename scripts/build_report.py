#!/usr/bin/env python3
"""Render the campaign's collected runs into the AutoArena results page."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.submission import comparison as report  # noqa: E402
from engine.records.collect import collect  # noqa: E402
from engine.records.runs import open_run  # noqa: E402
from engine.evaluation.task import load_task  # noqa: E402


def lane_names(plan: dict) -> list[tuple[str, str, str]]:
    """Every lane the campaign declares: the cross product plus `extra_lanes`."""
    names = [(m, t, f"{m}--{t}") for m in plan["methods"] for t in plan["tasks"]]
    names += [(e["method"], e["task"], e["name"]) for e in plan.get("extra_lanes", [])]
    return names


def summaries(plan: dict) -> list[dict]:
    base = ROOT / "runs" / plan["campaign"]
    out, skipped = [], []
    for method, task_id, name in lane_names(plan):
        directory = base / name
        if not (directory / "run.json").is_file():
            skipped.append(f"{name}: no run directory")
            continue
        try:
            summary = collect(open_run(directory), task=load_task(task_id))
        except Exception as error:                          # noqa: BLE001 - reported, not raised
            skipped.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if not (summary.get("launches") or summary.get("totals", {}).get("recorded")):
            skipped.append(f"{name}: nothing recorded")
            continue
        out.append(summary)
    for line in skipped:
        print(f"skipped {line}", file=sys.stderr)
    print(f"collected {len(out)} of {len(lane_names(plan))} lanes", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the collected summaries, so the page is auditable")
    args = parser.parse_args()

    plan = json.loads(args.campaign.read_text())
    collected = summaries(plan)
    if not collected:
        print("no lane produced a summary; nothing to render", file=sys.stderr)
        return 1

    out = args.out or (ROOT / "runs" / plan["campaign"] / "reports" / "results.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.build(collected, title=f"AutoArena results — {plan['campaign']}"))
    print(out)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(collected, indent=2, default=str))
        print(args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
