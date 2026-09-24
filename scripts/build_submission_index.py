#!/usr/bin/env python3
"""Build an index over exported submission directories."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.submission.export import write_json
from engine.submission.report import render_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    records = []
    for path in sorted(args.out.glob('*/submission.json')):
        record = json.loads(path.read_text())
        records.append({'name': path.parent.name, 'headline': record['headline'], 'verdict': record['verdict']})
    write_json(args.out / 'index.json', {'submissions': records})
    (args.out / 'index.html').write_text(render_index(records))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
