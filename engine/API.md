# Method API

AutoArena supplies the task, measures candidates and records results. Your method
owns proposals, search decisions, native logs and completion. It connects through
the Python client or the `autoarena` command-line client; both use the same
request and response contract.

While your method runs, API commands communicate with the engine through a local
Unix socket. The engine starts evaluation workers outside your method's sandbox.
A Linux sandbox may hide GPU devices while retaining the host network namespace
and the same absolute paths for the runner, context and candidate files. Do not
use a separate network namespace: the service uses a Linux abstract Unix socket.
No separate service command or method configuration is needed.

This reference covers `interface.json`, the supplied context, experiment requests,
responses and recovery. For operator commands, see the engine guide's
[run configuration and launch](README.md#start-a-run),
[resume and stop](README.md#lifecycle-and-evidence), and
[submission export](README.md#submissions). The root
[method integration walkthrough](../README.md#1-integrate-your-method) shows the
complete adaptation workflow.

## Method interface

Create `<your-method-repository>/interface.json`. Its directory is the method root
unless the run configuration explicitly overrides `source`. Supply the commands
that install dependencies, check readiness, start research and resume saved state.
The current interface is recorded internally as version 3. Earlier versions are rejected.

### Fields

| Field | Required | What you provide |
| --- | --- | --- |
| `id` | Yes | A stable method identifier, such as `my-method`. |
| `entry_type` | Yes | `"cli"` for a program entry point or `"prompt"` for a coding-agent entry point that consumes a prompt on stdin. |
| `environment.setup` | Yes | Installation commands as an ordered list of argument arrays. Use `[]` if no additional dependencies are needed. |
| `environment.check` | Yes | One command argument array that verifies imports, executable availability and any required services without starting research. |
| `start` | Yes | The noninteractive command and arguments that launch your program or coding agent. |
| `prompt_file` | For `prompt` | Relative path to a nonempty UTF-8 launch prompt inside your method repository. Omit for `cli`. |
| `resume` | Yes | Command and arguments that continue this same run from saved method state, without starting a fresh search. |
| `workers` | No | Parallel measurement capacity; defaults to one. Does not define search rounds or a research loop. |
| `version` | No | Omit, or explicitly use the current version, `3`. |

These are the complete interface fields. Keep method settings, dependency files and
supporting instructions in your own source files; your entry point or prompt reads
them. The runner records the supplied source's content identity automatically.

Every setup command, check, start and resume runs from the same private
copy at `.local/environments/<session_id>/source/` (or under `scratch_root`).
Relative paths such as `install.sh`, `check.py` and `run.sh` resolve there.
The engine verifies the initial copy against its recorded source hash before
setup. The copy persists, including setup-created files and a method-created
`.venv`; later edits to the original checkout do not affect the run.
Successful setup is reused. Before measurement starts, failed setup can be retried
explicitly with `python3 -m engine resume --run <run_dir> --retry-setup`; see the
[engine guide](README.md) for the command and retained logs.
Use `workspace` for candidate code and research records.

Methods own installation and activation. Each command is a separate process:
activating a venv or exporting a variable in setup does not activate it for check,
start or resume. Activate it inside each relevant script or invoke its interpreter
directly. For example, a method that creates `.venv` in `install.sh` can use this
`run.sh` (with its own `run.py`):

```bash
#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
exec python run.py "$@"
```

Before setup, the engine freezes `client/autoarena.py` under the environment root,
alongside `source/`. It adds that root's `bin/` to `PATH` for `autoarena` and
`client/` to `PYTHONPATH`.
These are supplied to setup, check, start and resume. No engine-created Python
environment is mandatory. Python `-I` ignores `PYTHONPATH`: if your method uses
isolated mode, install `autoarena-client` into that interpreter in your setup.
The installable client is packaged by the benchmark's root `pyproject.toml`.

Setup and readiness checks run without visible GPUs; failure stops the run before
its reference experiment. Do not install globally. Include any additional runtime
installation in `setup`. A `--help` readiness check only verifies CLI availability;
use a check script if your method needs services. Document required models/services
and credential variable names in your method's `README.md`, never credential values.

Commands are argument arrays, not shell strings. Common placeholders are:

| Placeholder | Supplied value |
| --- | --- |
| `{source}` | Private method copy, for every command. |
| `{task_dir}` | Frozen task definition and reference code. |
| `{workspace}` | Writable candidates, method state and native logs. |
| `{run_dir}` | Run records directory. |
| `{context}` | Run context JSON path; the file and `AUTOARENA_CONTEXT` are available at launch after reference measurement. |
| `{session_id}` | Stable session identity, retained on resume for native agent session arguments. |
| `{python}` | Runner interpreter, not an automatically selected method interpreter. |
| `{uv}` | Located `uv` executable for setup commands. |

Named run `bindings` may supply additional strings. Engine-owned values
cannot be overridden. Relative script paths and `AUTOARENA_CONTEXT` usually remove
the need for path placeholders.

### Entry types

Entry type selects **how we start your method**, not how it calls the benchmark.
Either entry type can use the [Python or command-line API](#experiment-api).
Both receive `AUTOARENA_CONTEXT` pointing to the same kind of context JSON file.

<details>
<summary><strong>entry_type: cli — run a program</strong></summary>

Your program reads the task context and runs its own search loop. The runner
executes `start` with empty stdin; it does not inject research instructions.
A Python script is also a CLI entry point:

~~~json
{
  "id": "my-method",
  "entry_type": "cli",
  "workers": 1,
  "environment": {
    "setup": [["bash", "install.sh"]],
    "check": ["bash", "check.sh"]
  },
  "start": ["bash", "run.sh"],
  "resume": ["bash", "run.sh", "--resume"]
}
~~~

Supply these scripts in your method root, including any activation they need.
For a method-installed executable, you may use `[".venv/bin/my-method"]` as `start`.
Your entry point may read `AUTOARENA_CONTEXT` or accept a path argument that you
add explicitly to `start`. Replace `--resume` with your program's actual arguments;
it must restore this run's saved state, not restart the search.

</details>

<details>
<summary><strong>entry_type: prompt — run a coding agent with your instructions</strong></summary>

Supply a real coding-agent launch command that accepts a prompt on stdin.
The runner freezes `prompt_file` and sends its exact UTF-8 contents to that command.
Your prompt supplies all instructions. Tell the agent to read the JSON file named
by `AUTOARENA_CONTEXT` for task and workspace information.

~~~json
{
  "id": "my-prompt-method",
  "entry_type": "prompt",
  "workers": 1,
  "environment": {
    "setup": [["bash", "install.sh"]],
    "check": ["bash", "check.sh"]
  },
  "start": ["bash", "run-agent.sh"],
  "resume": ["bash", "run-agent.sh", "--resume"],
  "prompt_file": "prompt.md"
}
~~~

Supply these scripts for your actual coding-agent runtime. `run-agent.sh` must
pass stdin to the agent and include its noninteractive flags and actual
session-resume arguments; use `{session_id}` in interface argv when needed.
Create `prompt.md` in your method repository. Describe your method's actual research
procedure, tell the agent to read the context JSON and task definition, and direct
experiments through the benchmark API. Keep settings in that prompt or method-owned
files it tells the agent to read.

On explicit resume, the runner uses your required `resume` command with the same
unchanged prompt and `AUTOARENA_CONTEXT`. Your resume command and native state
control continuation.

</details>

## Context

`AUTOARENA_CONTEXT` points to a JSON object with the fields
below. Paths are absolute paths on the local runner; read the supplied values
instead of constructing them. The engine measures the reference before starting
your method and includes its result here.

| Field | Type and meaning |
| --- | --- |
| `version`, `api_version` | Context version `5` and API version `1`, respectively; independent of interface version. |
| `run_dir`, `workspace` | Strings: run output directory and writable method workspace. |
| `task_dir` | String: frozen task directory containing `task.json` and `code/`. |
| `task` | Object: the complete frozen task definition, also available in `task_dir/task.json`. |
| `source`, `method` | Private method-copy path and the complete method interface object. Setup and launch use that copy; candidates and research state belong in `workspace`. |
| `bindings` | Object: named bindings supplied in the launch configuration, or `{}`. |
| `session_id` | String identifying this method session, retained on resume. |
| `runner_root` | String: working directory for invoking the supplied engine commands. |
| `reference_result` | Object with `candidate_id`, `launch_uuid`, `status` and `metrics`; example below. |
| `evaluate`, `finish`, `status`, `experiment_status` | Arrays of strings: executable and arguments for each operation, already bound to this run. The client uses these commands. |

For example, `reference_result` has this shape (numbers are illustrative for
`params`; metric names come from the selected task):

```json
{
  "candidate_id": "reference",
  "launch_uuid": "example-reference-launch",
  "status": "ok",
  "metrics": {
    "val_bpb": 1.04,
    "num_params_total": 50332176,
    "flops_per_token_measured": 239078400,
    "peak_vram_bytes": 47198976512,
    "training_data_tokens_available": 631241817,
    "total_tokens": 389545984
  }
}
```

Use this measured baseline in your research loop. The task's objective and
constraints determine how to compare it with candidates.

The context is available after reference measurement, when the method starts.
Initial setup and readiness checks must not depend on `AUTOARENA_CONTEXT` or its
JSON file.

### Task package

All nine supplied tasks have this layout:

```text
task_dir/
├── task.json              Objective, constraints, metrics and execution limits
└── code/
    ├── train.py           Starting model and training program
    ├── prepare.py         Data, validation and measurement functions
    ├── pyproject.toml     Training dependencies
    └── uv.lock            Exact dependency versions
```

`train.py` contains the model, optimizer, hyperparameters and training loop. It
imports the data and measurement functions from `prepare.py` and reports the
trained model through `report_efficiency_metrics(...)`. The dependency files
define the environment in which this code is measured. Your method's own
installation is configured separately in `interface.json`.

In the supplied tasks, `train.py` is editable and the other three files are fixed.
Editable files can still contain protected sections. Read both the file lists and
the code restrictions in `task.json` before choosing changes. Submit a complete
copy of `code/` as the candidate; keep the task definition and method logs outside
that candidate directory.

The full task definition is also available as `benchmark.task` in Python.
Read its objective, budget, file permissions and code restrictions before building
candidates. The [task field reference](../tasks/README.md#taskjson-field-reference)
explains every field, and the
[executable source contract](../tasks/README.md#executable-source-contract)
describes the required model and measurement interfaces.

## Experiment API

### Python client

`Benchmark.from_context()` reads `AUTOARENA_CONTEXT`. For example, at the point
where your method evaluates a candidate:

```python
from autoarena import Benchmark

benchmark = Benchmark.from_context()
# candidate_directory and experiment_id are supplied by your method.
result = benchmark.evaluate(
    source=candidate_directory,
    request_id=experiment_id,
    research_log={"ideas": ["Your method's proposal"]},
)
```

Supply a complete candidate directory and a saved, unique request ID. Feed
`result["metrics"]` and `result["status"]` into your method's existing feedback and
selection logic. Your method owns any conversion to native scores.

| Python API | Purpose |
| --- | --- |
| `Benchmark.from_context()` | Connect using the supplied context. |
| `benchmark.task`, `benchmark.task_dir`, `benchmark.workspace` | Read the task definition and locate task files and writable method state. |
| `benchmark.reference_result` | Access the already measured reference. |
| `benchmark.evaluate(source, request_id, research_log=...)` | Evaluate one candidate and return a result dictionary. |
| `benchmark.evaluate_batch(candidates, request_id)` | Evaluate a batch and return results in submission order. |
| `benchmark.experiment_status(request_id)` | Inspect an existing request without launching work. |
| `benchmark.status()` | Check run state and remaining budget. |
| `benchmark.finish(reason)` | Declare completion after processing the final results. |

`evaluate` and `evaluate_batch` wait for results. An experiment failure such as OOM
is returned in the result's `status`; request rejection or delivery failure raises
`APIError`. See [responses](#responses).

### Command-line client

The `autoarena` client reads `AUTOARENA_CONTEXT` and returns JSON. For example:

```bash
autoarena evaluate --source /absolute/candidate --request-id experiment-001 \
  --research-log '{"ideas":["Your method proposal"]}'
```

| CLI command | Purpose |
| --- | --- |
| `autoarena evaluate --source DIR --request-id ID --research-log-file FILE` | Evaluate one candidate with its research JSON. |
| `autoarena evaluate --batch FILE --request-id ID` | Evaluate a JSON array of candidates. Use `--batch -` to read stdin. |
| `autoarena experiment-status --request-id ID` | Inspect an existing request. |
| `autoarena status` | Return run state and remaining budget. |
| `autoarena finish --reason TEXT` | Declare completion after processing the final results. |

Paths supplied through `--source`, `--research-log-file` and `--batch` resolve
from the caller's working directory. Candidate `source` paths inside a batch
must be absolute. Request rejection or delivery failure produces a nonzero exit
code and an explanation on stderr. Experiment failures are returned in JSON.

### Batch requests

Each batch item supplies `candidate_id`, `source` and `research_log`.
`parent_id` and `idea` are optional metadata. A parent ID may be any nonblank
string naming a method-native parent, including one the benchmark has not measured.
An example request array is:

```json
[
  {
    "candidate_id": "candidate-001",
    "source": "/absolute/workspace/candidate-001",
    "research_log": {"ideas": ["Your method's proposal"]}
  }
]
```

Pass the array to `benchmark.evaluate_batch(candidates, request_id)` or save it
as the CLI's batch file. The engine assigns candidates to the interface's
configured `workers`; the response list preserves submission order. The method
chooses which candidates to submit and how to use their results.

### Research log

Before every candidate execution, supply a `research_log` JSON object containing
`ideas` (a nonempty list of nonempty strings or objects). `status` is optional; if
present, it must be a nonblank string describing your method's current state.
Additional JSON fields are allowed; all values must be JSON serializable and finite.
There is no required idea format or status vocabulary. This account does not replace
native logs, determine eligibility, or instruct the engine how to search.
`idea` and `parent_id` remain optional metadata.
For a large payload, the CLI accepts `--research-log-file /path/to/research.json`
instead of `--research-log JSON`. A batch supplies `research_log` in every item.

The engine validates every research payload before accepting a request, saves it in
the immutable dispatch `request.json` before starting its worker, and appends it to
the launch intent before execution. Missing or malformed logs reject the request
before candidate reservation or compute. The engine-owned initial reference is exempt.

### Request identity and recovery

Save a request ID and its research payload before submission. Reusing the same ID
and inputs recovers the existing response. Keep submitted candidate files and the
research payload unchanged; changing `research_log` under an existing request ID
is rejected.

Identical code cannot be submitted again while any attempt is charged or
unresolved. Once every attempt for that program is resolved and uncharged, the
method may submit it with new candidate and request IDs. An adjudication can
release the reservation; a refund without a result or explicit abandonment
leaves the attempt unresolved. Previous requests and ledger rows are preserved.

## Responses

### Evaluation result

`autoarena evaluate --source ...` and `benchmark.evaluate(...)` wait for the
evaluation and return one JSON object matching the
[evaluation schema](schema/evaluation.schema.json). Batch evaluation returns a
list of these objects in submission order. Example response:

```json
{
  "schema": "autoarena/evaluation/1",
  "run_id": "my-run",
  "task_id": "params",
  "method_id": "my-method",
  "launch_seq": 2,
  "launch_uuid": "example-candidate-launch",
  "candidate_id": "candidate-001",
  "status": "ok",
  "witness": true,
  "charged": true,
  "charge": "CHARGED",
  "charge_classification": "measured",
  "metrics": {
    "val_bpb": 1.03,
    "num_params_total": 47186446,
    "flops_per_token_measured": 213912576,
    "peak_vram_bytes": 42038909440,
    "training_data_tokens_available": 631241817,
    "total_tokens": 400000000
  },
  "extraction_errors": [],
  "logs": {
    "stdout": "/path/to/autoarena/runs/my-run/logs/0002/attempt-1/stdout.log",
    "stderr": "/path/to/autoarena/runs/my-run/logs/0002/attempt-1/stderr.log"
  },
  "benchmark_verdict_deferred": true
}
```

- `status` describes execution, such as `ok` or `crash`. An `ok` execution can
  still miss the task's quality or resource requirements.
- `metrics` contains the task's extracted measurements. Missing measurements
  are `null`; `extraction_errors` explains extraction failures. Read `logs` for
  training output and error details; these are local file paths.
- `witness` records whether the task's GPU-work marker appeared. `charged`,
  `charge` and `charge_classification` report the engine's budget accounting.
  A failed experiment can still consume a launch.
- `benchmark_verdict_deferred: true` means official eligibility is determined
  during collection and reporting using the task rules. Your method interprets
  the returned measurements for its own next decision.

### Errors and request status

An experiment crash is returned as a result. A rejected or undelivered API call
instead exits nonzero with an error on stderr; the Python client raises `APIError`.
Inspect `autoarena experiment-status --request-id ID` or
`benchmark.experiment_status(ID)` before recovering an interrupted request. It
returns `request_id` and `status`, plus `job_id`, `results` or `error` when available;
an unknown ID returns `{"request_id": "ID", "status": "not_found"}`. Recover using
the same request ID and unchanged inputs.

### Completion

`autoarena finish --reason "Search completed"` and
`benchmark.finish("Search completed")` return:

```json
{"method_declared_complete": true, "native_completion_verified": false}
```

This acknowledges your method's completion declaration. The engine does not
independently verify whether the method fulfilled its own research procedure.
Call it after processing your final results; evaluations must already be settled.
A successful process exit without `finish` leaves the run paused.
