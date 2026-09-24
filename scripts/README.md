# Setup and report scripts

- `prepare_local.py --task params` installs frozen measurement dependencies with
  Python 3.10 into `.local/measurement` and prepares `data/autoresearch` using a process-local mount.
  It requires `uv` and bubblewrap (`bwrap`).
- `build_submissions.py --lanes lanes.json --snapshots runs --out runs/submissions`
  exports the listed completed runs and an HTML index. Each lane needs a `name`.
- `build_submission_index.py --out runs/submissions` rebuilds the index.
- `leaderboard.py add --submission LOCAL_JSON --output site/submissions/METHOD/RUN`
  converts an engine export to the common website result format. Artifact links
  are optional; `--artifacts-url DOWNLOAD_URL` supplies code and logs.
  `check` validates result consistency offline;
  `build --output runs/leaderboard-site` renders the static public website with
   reviewed display names and source/license links from `site/methods.json`,
   and the blog's Elo calculation using the target families in `site/elo.json`.
  Checking and building include every `submission.json` inside method/run folders.
  See the [leaderboard guide](../site/README.md).
- `import_publication.py --organized DIRECTORY --output NEW_DIRECTORY` imports
  all organized results using the repository's method/target names and empty
  artifact objects.
- `upload_submission_reports.py --out runs/submissions --prefix PREFIX`
  uploads the complete directory using AWS credentials from the environment.
  Use a new project prefix; `--dry-run` lists uploads. Output URLs are unsigned
  and private objects require AWS authentication.
- `build_report.py --campaign FILE --out FILE` renders a campaign measurement
  comparison, with separate rankings per task.

See the [submission guide](../engine/README.md#submissions) for report contents.
