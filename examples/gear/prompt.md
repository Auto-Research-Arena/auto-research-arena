# GEAR on AutoArena

Read the supplied `AUTOARENA_CONTEXT`, `source/settings.json` and the complete
`source/upstream/program_gear_fixed.md`. Execute that upstream research loop in
this persistent coding-agent session. You own candidate implementation, reading
experiment results, debugging, postmeasurement reflections and controller feedback.
The integration below replaces the task execution and storage instructions.

## Task and execution

Use this run's `task_dir/task.json` and `task_dir/code/` as the task and reference.
Read the task's objective and complete constraints before proposing candidates.
The benchmark supplies dependencies, data, GPUs and the exact experiment duration.
The upstream five-minute estimate and ten-minute kill rule are replaced by the
registered task runtime. The evaluator manages its experiment processes.

Replace `uv run train.py` with `autoarena evaluate --source ABS_CANDIDATE
--request-id UNIQUE_ID --research-log-file ABS_RESEARCH_JSON`. Each submission
contains the complete declared task files. The research JSON contains your own
nonempty `ideas` list; a nonblank `status` is optional. The API returns raw results and log
paths; inspect those results and write the upstream postmeasurement reflection.
Canonical failures remain useful feedback. Changed-code crash fixes use the same
API and available task budget; request recovery retrieves the original experiment.

## Native controller and Git history

Create a fresh Git workspace containing the supplied reference code and a copy of
`source/upstream/gear.py`. Place it inside this run's own `workspace` directory from
`AUTOARENA_CONTEXT`, and put the API candidate directories there too.
Use the upstream Git commits, binding suggestions,
`record-run --apply-git`, native tables, reflections and artifact commits. Apply
the supplied constraint patch once to that workspace copy before `init` or
`record-run`:
`git -C ABS_NATIVE_WORKSPACE apply ABS_SOURCE/constraints.patch`. The pinned source
hash in settings identifies the original; the patched workspace controller has
its own hash in native records. Materialize only the task-declared files from a candidate
commit into a separate API candidate directory under that same `workspace`, and finish
writing them before you call the API: the evaluator reads `--source` when the request
arrives, so a directory that does not exist yet is a launch that cannot start.
Controller files and Git artifacts remain in the native workspace. Record the candidate commit with the API request ID.

The engine has already measured this run's reference. Record it once as the native
baseline instead of running it again. `python <source>/score.py` reads that original
result. For a child, save its canonical result JSON and use
`python <source>/score.py --result ABS_RESULT_JSON`. The helper only translates
observations into GEAR's controller fields; you call `gear.py record-run`
and write the scientific reflection yourself. Native commit IDs identify lineage;
the corresponding canonical baseline candidate ID is `reference`.

For every task, the helper divides the ranking target by this run's positive,
finite measured reference target for the controller's `val_bpb` slot. It supplies
peak VRAM / 2**30 as `memory_gb` and parameters / 1e6 as `params_m`.
An unobserved controller slot receives `0.0`; a missing declared controller
measurement produces `crash`. Actual language-model quality remains in
the raw `metrics.val_bpb`.
Pass the helper's `eligible` value as `--eligible true` or `--eligible false`, and
its `eligibility_reason` as `--eligibility-reason`, on each successful measurement's
`record-run` call (including the reference). The helper checks the frozen task's
quality gate and numerical constraints using the benchmark rules. An ineligible
measurement stays `--status ok`: it is recorded, cannot enter the elite population,
and earns no positive parent improvement reward. Eligible results follow the
original controller thresholds and promotion rules. Retain each helper output
with the experiment artifacts so actual metrics and constraint feedback remain
available alongside the normalized controller fields.

## Sequential research loop

Keep the native population size of four and execute experiments sequentially on
one GPU. Obtain one binding native suggestion, implement the candidate and evaluate
it through the API. Inspect the result, handle crashes as the native program
describes, write the reflection, call `record-run --apply-git` and commit the
artifacts before requesting the next suggestion from the updated controller.

The task's launch budget is the only limit. Keep requesting suggestions and
evaluating candidates until the benchmark refuses a further launch, then call
`autoarena finish` with your actual stopping reason. Do
not count your own experiment steps toward a target: charged launches include the
reference and every step recorded as a crash, so read `budget.max_launches` from the
frozen task and the spend from the benchmark rather than tracking a private tally.
Resume this same native session and Git history.
