#!/usr/bin/env python3
"""Upload built submission reports and print their plain object URLs."""

from __future__ import annotations

import argparse
import mimetypes
from pathlib import Path
import sys

BUCKET = "grp-r3d-intern-projs"
REGION = "us-west-2"

#: Every extension a submission directory contains. The logs and the reference source are
#: not optional extras: the page links them relative to itself, so uploading only the page
#: and the record publishes a report whose evidence 404s.
TYPES = {
    ".diff": "text/plain; charset=utf-8",
    ".zip": "application/zip",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".py": "text/plain; charset=utf-8",
    ".toml": "text/plain; charset=utf-8",
    ".lock": "text/plain; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="directory of built reports")
    parser.add_argument("--prefix", required=True, help="destination key prefix in the bucket")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    files = sorted(path for path in args.out.rglob("*")
                   if path.is_file())
    if not files:
        print(f"nothing to upload under {args.out}", file=sys.stderr)
        return 1

    import boto3

    s3 = boto3.client("s3", region_name=REGION)
    prefix = args.prefix.strip("/")
    urls = []
    for path in files:
        key = f"{prefix}/{path.relative_to(args.out).as_posix()}"
        content_type = TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not args.dry_run:
            s3.put_object(Bucket=args.bucket, Key=key, Body=path.read_bytes(),
                          ContentType=content_type)
        url = f"https://{args.bucket}.s3.{REGION}.amazonaws.com/{key}"
        urls.append((path.relative_to(args.out).as_posix(), url))
        print(f"{'would upload' if args.dry_run else 'uploaded'} {key}")

    index = next((url for name, url in urls if name == "index.html"), None)
    print(f"\n{len(urls)} object(s)")
    if index:
        print(f"index: {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
