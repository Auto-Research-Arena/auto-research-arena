# Evaluation

Loads task rules and performs canonical reference and candidate measurements.

| Module | Responsibility |
| --- | --- |
| [task.py](task.py) | Load and validate task definitions |
| [protocol.py](protocol.py) | Admit candidates, enforce the launch budget and schedule batches |
| [reference.py](reference.py) | Measure the reference and check its declared assertions |
| [execute.py](execute.py) | Check immutable files, execute one launch and record its outcome |
| [metrics.py](metrics.py) | Extract the task's declared measurements |
| [objective.py](objective.py) | Apply eligibility constraints and ranking |

[Task packages](../../tasks/README.md) supply objectives, limits and instruments.
[Compute](../compute/README.md) executes work; [records](../records/README.md)
stores and accounts for the outcomes.

Public single evaluations use the same batch path as multi-candidate requests.
Admission checks the task, candidate IDs, immutable files, research payload and
remaining launch budget before acquiring workers. `parent_id` is optional native
lineage metadata: when present and non-null it must be a nonblank string, but it
need not identify a measured candidate. Its text is preserved without path or
membership rules; candidate IDs still require safe identifiers.

After worker allocation, the evaluator checks the observed model of every selected
GPU against `task.launch.accelerator`, before writing any launch intent. Missing
observations or a mismatch refuse the batch and release the allocation. This check
also runs for prepared/resumed runs, so changing the inherited CUDA mask cannot
bypass the task requirement. Backends supply selected-device facts through
`gpu_names(ids)`; `task.accelerator_error` applies the shared task policy.

Execution receives the task's environment over the worker's configured defaults.
Source provenance comes from captured file identities and immutable-file checks.
The reference uses canonical execution; the runtime controls its initialization
and does not replay an interrupted reference. Batch intents precede execution,
so an exception after reservation can leave an unresolved, charged intent.

The raw response retains `charged`, `charge`, `charge_classification` and
`benchmark_verdict_deferred`. Official eligibility and ranking are applied
separately. Use the module-level objective functions; `Verdict` retains its
acceptance truth value. All six documented metric extractors and cross-checks
remain available even though shipped tasks use `metrics_json`.
