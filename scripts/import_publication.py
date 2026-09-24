#!/usr/bin/env python3
"""Copy organized publication results into the website's common JSON format."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.submission import leaderboard
from engine.submission.export import write_json


METHODS = {
    "arbor": "arbor",
    "autoscientist": "autoscientists",
    "gear": "gear",
    "heuresis": "heuresis-map-elites",
    "ours-beam": "beam-search",
    "ours-vanilla": "sequential-search",
    "tpe": "tpe",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for the converted website submissions")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new directory")
    catalog = json.loads((args.organized / "catalog.json").read_text())
    records = []
    for entry in catalog["runs"]:
        lane = args.organized / entry["path"]
        method_name, target = Path(entry["path"]).parts
        method = METHODS[method_name]
        target = "kvcache" if target == "kvcache-gpu" else target
        metrics = json.loads((lane / "metrics.json").read_text())
        provenance = json.loads((lane / "provenance.json").read_text())
        export = json.loads((Path(provenance["source_report"]) / "submission.json").read_text())
        with (lane / "rounds.jsonl").open() as stream:
            measurements = [json.loads(line)["record"] for line in stream if line.strip()]
        record = leaderboard.site_record(
            {"headline": metrics["headline"], "history": measurements,
             "cost": {**export["cost"], "budget": metrics["budget"]}, "setup": export["setup"]},
            method_id=method, task_id=target, run_id=f"{method}--{target}")
        leaderboard.project_site_record(record)
        records.append((method, target, record))
    pairs = [(method, target) for method, target, _ in records]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate method/target result")
    for method, target, record in records:
        destination = args.output / method / target
        destination.mkdir(parents=True)
        write_json(destination / "submission.json", record)
        write_json(destination / "artifacts.json", {})
    counts = Counter(method for method, _, _ in records)
    print(f"Imported {len(records)} runs and "
          f"{sum(len(record['measurements']) for _, _, record in records)} measurements.")
    for method, count in sorted(counts.items()):
        print(f"  {method}: {count} targets")


if __name__ == "__main__":
    main()
