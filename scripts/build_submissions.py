#!/usr/bin/env python3
"""Export completed runs and their submission index."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.submission.export import export, write_json
from engine.submission.report import render_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lanes', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    records = []
    for entry in json.loads(args.lanes.read_text())['lanes']:
        name = entry['name']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('lane names must be directory names')
        record = export(args.snapshots / name, args.out / name)
        records.append({'name': name, 'headline': record['headline'], 'verdict': record['verdict']})
        print(f'Exported {name}')
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'index.json', {'submissions': records})
    (args.out / 'index.html').write_text(render_index(records))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
