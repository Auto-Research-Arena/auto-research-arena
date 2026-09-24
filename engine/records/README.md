# Records

Stores run evidence, derives budget usage and verifies reported measurements.

| Module | Responsibility |
| --- | --- |
| [runs.py](runs.py) | Run metadata and paths to logs and captured source |
| [ledger.py](ledger.py) | Append-only launch records and adjudications |
| [charging.py](charging.py) | Decide whether a launch consumes budget |
| [collect.py](collect.py) | Fold recorded launches into results and summaries |
| [verify.py](verify.py) | Check measurements against logs, source and task rules |

[Submission](../submission/README.md) uses these records to build reports and exports.

`run.json` freezes the task and method interface at initialization. Launch logs
and `engine/source-snapshots/` supply the evidence for verification; mutable
method workspaces do not replace captured measurement sources. Lifecycle captures
the source snapshots; records resolves their paths.

`collect(run, task=...)` derives a summary from the ledger. Its `method` object
contains `method_id` and `title`, with the title taken from the captured interface's
`id` when available. Registry roles, deviations and publication flags are not part
of the result contract. See [result.schema.json](../schema/result.schema.json).
Budget and reference blocks are computed once and reused by integrity checks.

`verify(run, task=..., source_roots=...)` uses one folded ledger snapshot for
measurement, budget, reference, lineage and research-log checks. Source roots
default to the engine's captured sources. `Report.to_dict()` returns findings and
coverage counts; a passing report may still contain missing-evidence warnings.

Both functions accept `views=...` to reuse an already validated snapshot from
`run.views()`, as submission assembly does. `views=None` reads the ledger;
an explicit empty list remains empty. Verification still reads logs and captured
sources for the supplied views.

Corrections append an adjudication with a launch UUID, a `CHARGED` or
`NOT_CHARGED` decision and a nonempty array of string grounds. The canonical fold
validates this shape before using any correction. Decisionless rows and string
grounds are invalid; they are not normalized. The latest valid adjudication sets
the charge while preserving the result's classification and all recorded rows.
Only explicit `abandoned: true` settles a launch whose result cannot arrive.
See [launch.schema.json](../schema/launch.schema.json). Reads never rewrite
ledgers or measurements.
