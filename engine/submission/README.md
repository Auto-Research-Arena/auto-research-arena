# Submission

Builds reports and exports from recorded runs.

| Module | Responsibility |
| --- | --- |
| [data.py](data.py) | Gather recorded results, task definitions and audit evidence |
| [record.py](record.py) | Build the submission record and reconcile accounting |
| [export.py](export.py) | Write the offline submission directory, source and exact logs |
| [report.py](report.py) | Render submission pages and indexes |
| [comparison.py](comparison.py) | Render comparisons across multiple runs |
| [leaderboard.py](leaderboard.py) | Read uniform website result JSONs; build method/target tables, margin-aware Elo and curves |

```python
from engine.submission.export import export

export("runs/my-run", "reports/my-run")
```

Exports use the run's frozen task definition for eligibility and source identity.
Eligibility uses every recorded winning metric, independently of display columns.
Collection, verification, history and reconciliation share one canonical ledger
read; malformed ledger rows fail explicitly.

Source and log files retain their exact bytes. `changed_files` compares source
bytes; the readable diff notes changes that decoding cannot show. Missing stderr
is recorded as null with a caveat. Missing decisive stdout or source evidence,
failed measurement/accounting checks and blocking integrity findings still prevent
publication. Ordinary lineage warnings and grounded accounting disclosure notes
remain visible caveats.

`nonfinite_values` contains JSON Pointers into the final submission record for
values replaced by null. Headline metrics come from canonical measurements;
method-specific stdout decorations are available in the original logs. Optional
native usage metadata is included only when its shape is recognized.
Completion reports the recorded status, reason and process anomalies; directory
names are not treated as proof of scheduler requeues.

For multi-run reports, pass collected summaries to
`engine.submission.comparison.build`. It groups runs by task, preserves each run's
measured reference and excludes runs with real integrity blockers from ranking.
Method identity does not select a separate reference/control ranking path.

See the [submission command and artifact layout](../README.md#submissions).
For public PR submissions and the static website, see the
[leaderboard guide](../../site/README.md).
The website uses `autoarena/site-submission/1` records containing measured
results, budget counts and compute/model metadata. `scripts/leaderboard.py add`
converts an engine export to that format, keeping exactly the target's declared
metric fields and using `null` for unavailable values. Website checks verify the
metric fields and that values and counts agree within the record; artifact
download links are optional.

Website comparisons use equal target weights and the ordering and rounding steps
in `site/elo.json`. Training-time display values are rounded to 30-second
checkpoints; source submissions retain their raw measurements. Elo averages
4,000 shuffled passes with a cohort mean of 1500 and exports full-precision
ratings and match-order standard deviations.
