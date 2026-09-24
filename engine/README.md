# Run engine

Start a method, give it a task and an evaluator, and let it own the research.
The engine does not define rounds, roles, proposals, selection or internal
thresholds. A method can be a program or a prompt-driven coding agent.

New method? Start with **[Bring your method to AutoArena](../README.md#1-integrate-your-method)**: connect your
experiments to our API and supply your installation and launch configuration. This page
is the detailed command and configuration reference.

## Code organization

| Package | Responsibility |
| --- | --- |
| [runtime/](runtime/README.md) | Configuration, method setup, process lifecycle and request dispatch |
| [evaluation/](evaluation/README.md) | Task contracts, reference/candidate measurements, metrics and eligibility |
| [records/](records/README.md) | Run storage, launch ledger, budget accounting, collection and verification |
| [submission/](submission/README.md) | Submission records, exports and rendered reports |
| [compute/](compute/README.md) | GPU allocation, subprocess execution, cache paths and dataset mounts |

`__main__.py` provides the CLI. `autoarena.py` is the standalone method client;
`schema/` contains the data formats. Each package README lists its core modules.

## Method interface

The method supplies an `interface.json` in its source root. It declares setup,
readiness, start and resume commands, an entry type, and measurement-worker capacity.
Prompt entries also supply their launch prompt. See the [method API](API.md#method-interface)
for the complete fields, command placeholders and examples.

Setup, readiness, start and resume use the same private method copy. The method
owns dependency installation and activation; successful setup is reused on resume.
The engine provides the Python and command-line client. `AUTOARENA_CONTEXT` becomes
available after the reference measurement, when research starts.

## Start a run

The benchmark operator creates this configuration and launches the run; the method
submitter supplies its source and interface. See the [launch walkthrough](../README.md#2-prepare-for-a-run)
for the complete sequence and the split of responsibilities.

From the repository root, create a run config with repository-relative paths:

```json
{
  "task": "params",
  "method": "examples/my-method/interface.json",
  "run_dir": "runs/my-run",
  "research_llm": {"used": false},
  "compute": {"backend": "local", "visible_devices": [0]}
}
```

The method root defaults to the directory containing `interface.json`; an optional
run-config `source` overrides it. Include code, installation files and any launch
prompt. The engine records the source hash and verifies its initial private copy
before setup. Installation and execution use that copy; later edits to the original
checkout do not affect the run. Keep the original source separate from `run_dir`,
which must not already exist. Named `bindings` are optional.
Replace GPU 0 with devices allocated to this run; separate local runs need disjoint GPUs.

The example above declares no LLM. For an LLM method, record its provider/model
and optional probe in `research_llm`; this does not choose the agent or rewrite
launch commands. The root walkthrough's [GEAR interface](../examples/gear/interface.json)
explicitly selects Opus 5 (`us.anthropic.claude-opus-5[1m]`), retains `{session_id}` for continuation and runs
`["python", "check.py"]` relative to the private method root. Configure native
permissions and inherited authentication before launch.

The operator supplies a working [local GPU runtime](compute/README.md), GPU capacity
and benchmark task data/runtime. The runner requires Python 3.10+; `uv` is used
for benchmark preparation; methods install their own runtimes in setup.
An optional run-config `scratch_root` selects local storage; the default is
`.local/`. Method environments, downloads and caches
live there, under a unique run identity. Setup commands are noninteractive, have
no visible GPUs, and each has a 30-minute timeout. Runs use local GPUs.
Secrets never go in config, arguments or support files.

```bash
python3 -m engine check --config run-config.json
python3 -m engine run --config run-config.json
```

`check` prints what was checked and whether it is ready. Without arguments it
checks task packages and method interfaces. With `--config` it checks the selected
API interface, method source/prompt and recipe, local GPU capacity, dataset/mount
dependencies and a CUDA operation in the selected measurement runtime. It renders
all method commands with the same binding logic as `run`, without executing installation
or starting research. Optional `research_llm.check` runs a short model probe.
`--json` returns the same checks as JSON. Checks do not purchase benchmark evaluations.

`run` freezes inputs, checks compute readiness, prepares the private method copy
and client, runs setup and the method's readiness command, and checks start/resume
entry points. Only then does it measure the reference and start the method from
the private method root. `runs/my-run/workspace/` holds candidates and research state.

Setup output is retained in `engine/setup/`; a successful environment receipt is
recorded.
After an installation or readiness failure, fix the dependency issue or the setup
script in the private method copy, then explicitly retry:

```bash
python3 -m engine resume --run runs/my-run --retry-setup
```

This reruns the configured setup commands in the existing copy and saves new logs
under `engine/setup/retry-*/`. It is available only before measurement starts.
After successful setup, the engine measures the reference and uses `start` for
the first controller launch. Ordinary resume reuses installed dependencies,
verifies the client and reruns readiness without replaying setup.

The process receives `AUTOARENA_CONTEXT`, a path to frozen JSON with the task,
reference result, method launch configuration, paths and evaluation commands. Read `task_dir`,
`workspace` and `run_dir` from this JSON, not separate environment variables.
Reference code is at `task_dir/code/`; `reference_result` holds its measured baseline.
With `entry_type: cli`, stdin is empty and the program reads the supplied context.
With `entry_type: prompt`, stdin contains only the method's frozen `prompt_file`
contents, unchanged on both start and resume. The supplied coding-agent command
consumes that prompt; `AUTOARENA_CONTEXT` supplies the context JSON path. Your
`resume` command controls continuation. An explicit entry type and a resume command
are required.

## Evaluate and finish

A launched method uses the [Python or command-line client](API.md#experiment-api)
to submit complete candidate directories. For example:

```bash
autoarena evaluate --source "$CANDIDATE_DIR" --request-id experiment-001 \
  --research-log '{"ideas":["Your method proposal"]}'
```

The method saves each request ID and research payload before submission, processes
the returned measurements in its own loop, and calls `autoarena finish --reason TEXT`
after its final feedback. Batch requests use the configured measurement workers.
Budget exhaustion leaves the controller running so it can save that feedback.

See the API reference for [batch requests](API.md#batch-requests),
[research logs](API.md#research-log), [recovery](API.md#request-identity-and-recovery)
and [response formats](API.md#responses). `finish` requires all accepted evaluations
to be settled. A successful process exit without it leaves the run paused.

## Lifecycle and evidence

```bash
python3 -m engine status --run runs/my-run
python3 -m engine resume --run runs/my-run
python3 -m engine stop --run runs/my-run
python3 -m engine collect --run runs/my-run
python3 -m engine verify --run runs/my-run
```

This Linux foreground supervisor has exclusive controller ownership. Resume is
explicit and preserves task, source, settings and native workspace. Only settled
paused/failed exits can resume. Interrupted ownership, an unfinished reference or
ambiguous evaluation needs recovery, not a blind restart. Completed and
operator-stopped runs cannot be reopened.

Stop prevents new requests and signals only the owned controller group. Already
accepted durable evaluations drain and retain evidence. Stop does not terminate
another job or release externally provisioned compute. Methods must clean up their
own nested sessions/services; this is not a sandbox or a host-wide process sweeper.
The optional `examples.claude_transport` helper handles its owned native CLI group. Process stdout and
stderr are retained, so controllers must not print secrets.

Useful files are created when consumed:

```text
run.json, launches.jsonl       frozen identity and append-only measurements
engine/definition.json        interface, bindings and method/task source hashes
engine/context.json           handoff consumed by the method
engine/task/                  frozen task.json and reference code/ package
engine/setup/                 setup/check output
engine/environment.json       successful environment receipt
engine/attempt-*/              controller input and process logs
engine/source-identities/     task-program reservations
method/dispatch/               durable evaluation requests and responses
engine/source-snapshots/      exact declared source captured for each evaluation
candidates/reference/         initial reference working copy
workspace/                    candidates, method-created research state and helpers
```

Collection reports measurements, ranking and audit findings. Use the submission
export below for the complete report and its supporting artifacts.

## Submissions

Export one completed run:

```bash
python3 -m engine submission --run runs/my-run --output runs/my-run/submission
```

The public JSON contract is [submission.schema.json](schema/submission.schema.json).
It defines the exported result, task constraints, compute/model setup, evaluation
history, source references and evidence.

The output directory must be new. The exporter reads the run without changing
its ledger or measurements. Open `submission.html` and share the entire directory:

```text
submission.html               offline report
submission.json               machine-readable record (autoarena/submission)
code/best/                    exact winning source and declared dependency files
code/reference/               exact evaluated reference source
code/changes.diff              reference-to-winner changes
code/best.zip                  source, diff and reproduction bundle
reproduce/task.json           frozen task definition
reproduce/evaluation.json     entrypoint, launch settings and source identity
reproduce/README.md            reproduction steps
evidence/history.json         every evaluation and recorded research account
evidence/ideas/                readable full research accounts, linked from history
evidence/logs/                 evaluation stdout and stderr
evidence/source-manifest.json measurement-time source hashes
evidence/log-manifest.json    log paths, SHA-256 hashes and byte counts
evidence/audit.json            audit findings
evidence/reconciliation.json  accounting checks
```

The summary shows the best qualifying result, reference and improvement. Setup
shows the recorded research model/provider, GPU instance/model, configured workers
and GPUs per evaluation. Evaluation GPU capacity is `workers × GPUs per evaluation`
from launch configuration. GPU-hours sum recorded evaluation durations multiplied
by GPUs per evaluation. Available controller usage records retain their reported
scope. The constraints table contains only the frozen task's gate and constraints.

Each evaluation shows its result and recorded idea, with a link to the full
method-authored research account. Code is verified against its measurement-time
identity before export. Evaluations capture the task's declared source files
in `engine/source-snapshots/<candidate_id>/`; execution continues in the submitted workspace.
The winning implementation includes the source itself, reference, diff and hashes.

The export includes the run's completion status and audit findings. A qualifying
submission needs a completed run, a valid reference and winner, consistent
measurement evidence and exact winning/reference source. Other evaluation evidence
gaps remain visible in the audit. Reporting uses the run's frozen task rules.
The export's `verdict.publishable` reports whether its publication checks passed.
Deployment configuration is excluded from the setup summary. Exported text retains
its recorded paths and identifiers; log and source files retain their exact bytes.
Each log-manifest entry contains its exported `path`, `sha256` and `bytes`.

For several completed runs, `lanes.json` lists their directory names:

```json
{"lanes": [{"name": "my-method--params"}, {"name": "my-method--kvcache"}]}
```

```bash
python3 scripts/build_submissions.py --lanes lanes.json --snapshots runs --out runs/submissions
```

This creates one report directory per run and an `index.html`. To rebuild only
the index, use `python3 scripts/build_submission_index.py --out runs/submissions`.

The Python entry points are `engine.submission.record.build(run_dir)` for a record,
`engine.submission.export.export(run_dir, output)` for the complete directory, and
`engine.submission.report.build(record)` for HTML. No LLM call or evaluation is
performed by export. The page uses escaped method text and carries no external assets.

## Validation

[Evaluation](evaluation/README.md) loads task definitions, extracts measurements
and ranks eligible results. [Records](records/README.md) implements accounting and
evidence verification. Format schemas live in `schema/`.
Task definitions use explicit `target`, `quality_gate`, `constraints` and `tiebreaks`
fields. The loader accepts this format only. See the
[detailed task field reference](../tasks/README.md#taskjson-field-reference);
`tasks/` itself contains task packages and their index.

CPU tests exercise the public client and CLI through real subprocesses with fake
GPU measurements: task handoff, exact feedback, required research logs and optional lineage,
request recovery, pause/resume, ownership and canonical audit. They test the API,
not the research quality or fidelity of a particular method. Baseline native-run
validation is separate from this contract.

## Local storage

Launch configuration paths are relative to the repository root. `.local/` holds
method environments, dependency caches and scratch; `data/` holds datasets; `runs/`
holds evaluation records and reports. These directories are ignored by Git.
Prepare the shared measurement runtime and data with:

```bash
python3 scripts/prepare_local.py --task params
```

The preparation script requires `uv` and bubblewrap (`bwrap`) for its dataset mount.
It installs the shared Python 3.10 measurement runtime in `.local/measurement`
and prepares data/tokenizer files in `data/autoresearch`.

Local run configs default `compute.env` to repository cache paths,
`UV_PROJECT_ENVIRONMENT=.local/measurement` and `UV_NO_SYNC=1`; explicit environment
entries override these defaults. `compute.mounts` defaults to `{"~/.cache/autoresearch": "data/autoresearch"}`.
An explicit mount map replaces the default; `"mounts": {}` disables it.
The local worker uses `bwrap` to apply mounts only to its subprocess, preserving
the task's frozen source bytes and the host's existing cache. Method installation
and measurement dependencies remain separate.

### Research LLM metadata and probe

A launch can declare `research_llm` with `provider`, `model`, `settings`, and an
optional `check` command argument list. The command should request a short model
response; `engine check --config` reports success only when it exits successfully.
These are metadata and a probe only: the engine does not infer an agent or model
command, or change `start`/`resume` from them. Select the actual runtime and model
in the method's commands or scripts. Credentials remain inherited from the shell.
The report includes model metadata and available usage records, and excludes the
probe command. For a method without an LLM, use `"research_llm": {"used": false}`.

Readiness output names each check and shows `READY`, `FAILED` or `SKIP` with its
scope. A skipped model probe does not claim LLM availability. Use `--json` for
machine-readable output. Method-specific installation and its environment check
run before reference evaluation.
