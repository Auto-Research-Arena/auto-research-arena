# Submitted results

Keep each method's runs together:

```text
site/submissions/
  beam-search/params/
    submission.json
    artifacts.json
  sequential-search/params/
    submission.json
    artifacts.json
```

Every `submission.json` uses `autoarena/site-submission/1`. It contains the
method and target IDs, winner and reference values, recorded budget usage,
compute/model metadata, and measurements. Each measurement has its launch
sequence, candidate ID, reference/charge/qualification flags, and metrics.
Metric fields exactly match `metrics` in the target's `task.json`; unavailable
values are `null`. Extra instrument diagnostics are omitted.

`artifacts.json` is `{}` until download links are supplied. Supported fields
are `code_url`, `logs_url` and `report_url`; the page shows only available links.

Use `python3 scripts/leaderboard.py add --submission ENGINE_EXPORT_JSON
--output site/submissions/METHOD/RUN` to convert an engine export. Add
`--artifacts-url URL` when the code and logs archive is available.

Check all records with `python3 scripts/leaderboard.py check`, then build with
`python3 scripts/leaderboard.py build --output runs/leaderboard-site`.
See the [website guide](../README.md) for the full workflow.
