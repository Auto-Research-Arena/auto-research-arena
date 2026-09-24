# TPE integration changes

Compared with [autoresearch-automl](https://github.com/ferreirafabio/autoresearch-automl):

| Part | Original | This integration |
| --- | --- | --- |
| Task | Training-file path and validation-loss objective. | [adapter.py](adapter.py) reads the task and reference; [tpe_scoring.py](tpe_scoring.py) reads targets and constraints. |
| Search space | Extracts tunable parameters from the training file. | A name filter selects fourteen hyperparameters and leaves task budget constants fixed. |
| Trial duration | Configurable range, default 60–300 seconds. | Both endpoints use the task's fixed training duration. |
| Initial trial | Measures the starting configuration. | Uses task reference parameters and the already measured reference. |
| Candidate files | Edits a job-specific script. | Applies the same parameter editor to a complete task-code copy per candidate. |
| Execution | Training subprocess. | Saves parameters, status and request identity, then calls `Benchmark.evaluate`. |
| Feedback | Validation loss. | Reference-normalized target with shared constraint penalties, with the canonical result attached. |
| Resume | Replays trial records and may redraw an interrupted trial. | Identical code reuses its candidate/request ID. Different code under the same trial number gets a new ID, preserving earlier files and requests. |
| Stopping | Configurable trial limit. | Defaults to 99 candidate trials plus the engine reference. |
| Sampler startup | Optuna's default ten random startup trials, against hundreds of trials. | The native default of ten startup trials. |

Every task uses the same score in [tpe_scoring.py](tpe_scoring.py); lower is better.
The target, quality gate and constraints come from `task.json`.

| Outcome | Score |
| --- | --- |
| All requirements satisfied | Measured target / this run's measured reference target |
| Only the quality gate violated | `1000 + abs(quality - gate threshold)` |
| Another constraint violated or its measurement missing/nonfinite, or target nonpositive | `2000` |
| Execution unsuccessful, or target or quality measurement missing or nonfinite | `3000` |
| Otherwise admissible, but reference target missing, nonpositive or nonfinite | `3000` (`no_reference`) |

An execution or target/quality measurement failure takes precedence over constraint
penalties; another constraint violation takes precedence over a quality-gate violation.
The target ratio uses no logarithm, clipping or fixed reference constant.
These scores occupy the native `val_bpb` field; actual quality and all other
canonical measurements remain in `extra_metrics.canonical_result`.

[runtime.patch](runtime.patch) lets the native runner accept the API experiment
executor. Installation applies the patch to a private source copy.
