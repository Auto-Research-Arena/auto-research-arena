# Benchmark tasks

Nine standalone task packages. Each contains:

- `task.json`: the exact pre-registered machine-readable definition.
- `code/`: runnable reference code and dependency lock.

For method integration, read [what a task package supplies and how to interpret its JSON and code](../README.md#11-task-definitions).

| Task | Minimize |
| --- | --- |
| [data](data/task.json) | Total training tokens, including repetitions, to the first passing quality evaluation |
| [flops](flops/task.json) | Counted forward/backward FLOPs per token |
| [memory](memory/task.json) | Process-wide peak allocated device memory through reporting |
| [params](params/task.json) | Total model parameters |
| [steps](steps/task.json) | Optimizer updates to the first passing quality evaluation |
| [traintime](traintime/task.json) | Training time to the first passing quality evaluation |
| [decode](decode/task.json) | Median request latency with 1 prefill token and 512 decode steps |
| [kvcache](kvcache/task.json) | Incremental allocated state memory with 1,536 prefill tokens and capacity 2,048 |
| [request](request/task.json) | Median request latency with 1,536 prefill tokens and 512 decode steps |

`objective.target` names what to optimize. `quality_gate` and `constraints` state
what a result must satisfy; `tiebreaks` apply when target values are equal.

The quality gate and resource constraints determine official eligibility, independently
of how a searcher chooses intermediate candidates. Each task has a fixed reference,
budget and measurement protocol. Use the evaluator instead of launching training outside
its accounting. See [the engine](../engine/README.md).

Candidate code must respect the semantic rules in `substrate.known_constraints` and
the protected regions in `substrate.frozen_regions`. These assume cooperative agents;
immutable-file checks do not certify every behavior.

The loader resolves `code/` next to each definition. Code shared by several tasks is
copied intentionally so each folder is usable without a second code tree. Definitions
and declared files are checked by `python3 -m engine check`. History remains in Git;
previous runs retain their frozen definitions.

Shared loading, validation, metric extraction, accounting and ranking live in
`engine/`. This directory contains task packages, not engine implementation.

## `task.json` field reference

This reference covers every field supported by the current task format. Read it alongside the
[JSON Schema](../engine/schema/task.schema.json) and
[semantic loader](../engine/evaluation/task.py). The schema describes the structural
contract; the loader performs its own checks rather than running a JSON Schema
validator. Schema acceptance alone does not establish executable behavior.
Required fields have no implicit default unless one is explicitly stated below.
Optional metadata does not supply a measurement when omitted.

Paths use `[]` for an array element and `<metric>` for a key in the `metrics`
dictionary. For example, `metrics.<metric>.unit` describes
`metrics.val_bpb.unit` as well as every other metric's unit. Field names in
metric references must match dictionary keys exactly.

- [Identity and top-level objects](#identity-and-top-level-objects)
- [Substrate and protected source](#substrate-and-protected-source)
- [Launch settings](#launch-settings)
- [Budget and charging](#budget-and-charging)
- [Metric declarations and extractors](#metric-declarations-and-extractors)
- [Metrics used by the nine tasks](#metrics-used-by-the-nine-tasks)
- [Objective, eligibility and comparison](#objective-eligibility-and-comparison)
- [Reference launch](#reference-launch)
- [Reporting](#reporting)
- [Executable source contract](#executable-source-contract)
- [Task descriptions](#task-descriptions)

### Identity and top-level objects

| Field | Type / presence | Meaning |
| --- | --- | --- |
| `task_id` | Required string | Stable task identifier, matching the containing directory name. The schema permits lowercase letters, digits and hyphens, starting with a letter or digit. Artifacts identify the task by this value; it does not select a search method. |
| `version` | Required integer, at least 1 | Definition version recorded with the task. All nine definitions use `1`. This is separate from an instrument's output-schema version. Changing an objective or gate requires a new task ID, not merely incrementing this field. |
| `title` | Required nonempty string | Human-readable task name, carried into collected results. It does not define the ranking arithmetic. |
| `summary` | Optional string | Descriptive metadata, absent from the nine current tasks. No execution or scoring effect. |
| `supersedes` | Optional string or `null` | Prior task ID for definition history. All nine tasks name their predecessor. This neither loads that predecessor nor imports its measurements. |
| `substrate` | Required object | Source provenance, preparation/execution commands, editable and immutable files, and semantic restrictions. |
| `launch` | Required object | Per-measurement execution conditions; distinguish declarations from settings passed to the worker as described below. |
| `budget` | Required object | Campaign allowance in charged launches and the accounting rule. |
| `metrics` | Required nonempty object | Dictionary from metric names to typed extraction specifications. It defines which output quantities the engine reads. |
| `objective` | Required object | Eligibility and comparison rules. Contains `comparison_mode`, `target`, `quality_gate`, `tiebreaks` and `constraints`. |
| `reference_launch` | Required object | Required initial reference measurement and deterministic reproduction assertions. |
| `report` | Required object | Headline and ordered metric columns for collected results. |

The schema rejects unspecified fields in these objects and in most nested
records. The intentional open dictionaries are metric names, environment
variable names and exact-assertion metric names.
The semantic loader additionally rejects unresolved placeholder strings and
invalid cross-references. Preserve published definition and instrument bytes;
this guide is not a second specification.

### Substrate and protected source

| Field | Type / presence | Meaning and implementation |
| --- | --- | --- |
| `substrate.id` | Required string | Substrate provenance label. The training tasks use `karpathy-autoresearch-v1`; serving tasks use `karpathy-autoresearch-v1-inference`. The engine uses the declared contract rather than branching on this label. |
| `substrate.source` | Required string | Upstream source location for provenance. Loading a task does not clone this location: execution uses the packaged `code/` directory. |
| `substrate.revision` | Required string | Recorded upstream revision. Current packages record `716857590764cc458562d3483a7279fc6ae6edb6`. This is provenance for the substrate, not a claim that the instrumented package is an unchanged upstream checkout. |
| `substrate.prepare` | Required nonempty array of strings | Preparation command as an argument vector, currently `["uv", "run", "prepare.py"]`. Each element is one argument. This documents how to prepare task data; the current reference/evaluation lifecycle does not automatically run this command. |
| `substrate.entrypoint` | Required nonempty array of strings | Experiment command as an argument vector, currently `["uv", "run", "train.py"]`. The evaluator passes it to the worker with the candidate source directory as its working directory; it is not a shell command string. |
| `substrate.mutable` | Required nonempty array of strings | Candidate-editable package paths, currently `["train.py"]`. Protected regions and semantic restrictions still apply inside editable files. This list must not overlap `immutable`. |
| `substrate.immutable` | Required array of strings; may be empty structurally | Protected package paths, currently `prepare.py`, `pyproject.toml` and `uv.lock`. Before launch, the evaluator compares existing candidate files against the pinned bytes and refuses edits. Package checking requires every declared file to exist. The evaluator's byte comparison skips missing files; it is not a complete source-tree compliance check. |
| `substrate.gpu_work_witness` | Required nonempty string | Literal substring sought in this launch's captured stdout to establish chargeable GPU work. All nine use `Parameter counts:`, printed after model allocation. It is neither a regular expression nor a GPU utilization measurement. |
| `substrate.frozen_regions` | Optional array of objects; absent acts as `[]` | Protected behavior inside otherwise mutable files. Every current task supplies this list. These are declarations for participants and source audits, not automatic syntax-tree matching. |
| `substrate.frozen_regions[].file` | Required nonempty string | Package-relative file containing the protected region, currently `train.py`. |
| `substrate.frozen_regions[].region` | Required nonempty string | Human-readable name of the protected operation or block. |
| `substrate.frozen_regions[].anchor` | Optional nonempty string | Source locator to help a reader find the operation. Its presence is validated as text; matching the anchor is not an executable compliance test. |
| `substrate.frozen_regions[].reason` | Required nonempty string | Why the region is protected and what must be preserved, including the arguments and placement of measurement calls. Read the complete text, not only the region label. |
| `substrate.known_constraints` | Optional array of nonempty strings; absent acts as `[]` | Semantic candidate restrictions and measurement scope. Every current task supplies them. They cover truthful reporting, data access, clocks, counters, countable arithmetic and, for serving tasks, decode-state behavior. They are distinct from the numeric clauses in `objective.constraints`. |

`python3 -m engine check` checks that declared mutable/immutable files are regular
package paths, exist, and are not symlinks or paths escaping the package.
The [evaluator](../engine/evaluation/execute.py) enforces the immutable byte boundary.
Neither this check nor the prose declarations authenticate every action of
candidate Python code.

### Launch settings

| Field | Type / presence | Meaning, units and current behavior |
| --- | --- | --- |
| `launch.train_seconds` | Required positive number | Declared training-clock budget in seconds, `600` in every task. The actual loop uses the frozen `TIME_BUDGET` constant and its stop condition. The evaluator does not turn this JSON field into a timer or rewrite source constants. |
| `launch.gpus_per_launch` | Required integer, at least 1 | Number of GPUs allocated to one evaluation, `1` in every task. Used to divide local devices among workers and included in each launch request. It is not the number of candidate evaluations or research workers. |
| `launch.accelerator` | Required nonempty string | Declared measurement hardware, `NVIDIA_A100_80GB` in every task. Local preflight checks the accelerator in the compute configuration against visible GPUs, not this task field. |
| `launch.seed` | Required integer, including an explicitly supplied `0` | Declared reference seed, `42` in every task. The pinned training scripts seed PyTorch and CUDA directly with `42`; the evaluator does not inject this field into arbitrary candidate code. |
| `launch.timeout_seconds` | Required positive number, greater than `train_seconds` | Worker process timeout in elapsed seconds, covering startup, compilation, training, validation and reporting. It is `2700` for `data`, `steps`, `traintime`, and `1800` for the other six tasks. Timeout termination can add a grace period; this is separate from the training clock. |
| `launch.env` | Optional object mapping variable names to strings; absent supplies no task overrides | Environment entries passed to every launch, including the reference. Merge precedence is backend defaults, then task entries, then low-level evaluator caller entries. This merge does not enforce immutability; the task's measurement settings remain contractual. |
| `launch.env.<variable>` | String value for each supplied key | Literal process-environment value, not an arbitrary JSON number or boolean. The instrument determines whether and how to parse it. |
| `launch.env.AUTORESEARCH_TARGET_VAL_BPB` | String, present in the three to-target tasks | `"1.05"` in `data`, `steps` and `traintime`. Their training script parses it as a float to construct the frozen harness. The harness reports `target_val_bpb`, which is pinned by an equality constraint. The other six tasks omit this setting and declare that no harness is used. |

See [launch construction](../engine/evaluation/execute.py),
[local worker execution](../engine/compute/runner.py) and
[backend preflight](../engine/compute/backend.py) for these execution settings.

### Budget and charging

| Field | Type / presence | Meaning and supported behavior |
| --- | --- | --- |
| `budget.max_launches` | Required positive integer | Maximum charged launches for the run, `100` in every task. Reference and candidate evaluations share this allowance across workers. Admission counts existing charged or unresolved launches before accepting more work. |
| `budget.includes_failures` | Required boolean; loader requires `true` | Failures with a GPU-work witness consume budget. It does not mean every attempted command is charged: failures without that witness are classified by the charging rule. `false` is rejected, even though the schema's type alone allows it. |
| `budget.includes_reference_launch` | Required boolean; `true` in all nine tasks | Declares reference inclusion and is reported/audited. The loader accepts `false`, but the implemented charging/admission logic still counts witnessed reference launches. `false` is not an implemented free-reference mode. |
| `budget.charging_rule` | Required string, exactly `"gpu-work-witness"` | The only implemented charging rule. Reads recorded execution evidence, not objective values, elapsed GPU seconds or the method's opinion of a failed experiment. |

The [charging rule](../engine/records/charging.py) applies to the effective result of
each distinct launch, using the append-only ledger and any adjudications:

| Evidence | Charge |
| --- | --- |
| Intent exists, result unresolved | Charged conservatively while the outcome is unknown. |
| Result has the stdout witness | Charged, including crashes, out-of-memory failures, timeouts and interruptions, regardless of a missing exit code. |
| Result has no witness and no exit code | Not charged; classified as never executed. |
| Result has no witness but has an exit code | Not charged; classified as a substrate failure. |

The witness is tested before the exit code. Eligibility and charging are
separate: an ineligible measured candidate can still consume one launch.
A normal charged reference leaves `99` of the `100` launches for candidates.
Changing a candidate/request label does not authorize another measurement of
the same program; the execution protocol also records program identity.

### Metric declarations and extractors

Each `metrics.<metric>` is an object describing a quantity and its extraction.
Declaring a metric makes the engine read it; only objective clauses give it a
role in eligibility or ranking. Reporting and reference assertions are separate
uses of the same extracted value.

| Field | Type / presence | Meaning |
| --- | --- | --- |
| `metrics.<metric>` | Object for each declared metric | Specification keyed by the canonical metric name. The name need not equal the output field name; `extract.field` supplies that mapping. |
| `metrics.<metric>.type` | Required string: `integer`, `number` or `boolean` | Expected scalar type. The extractor coerces supported output values; an integer read must be integral within `1e-9`. Boolean text accepts `true`/`yes`/`1` and `false`/`no`/`0`. Missing values remain `null` regardless of this declaration. |
| `metrics.<metric>.unit` | Optional string; no conversion/default | Human-readable unit. For example `bytes`, `ms`, `s`, `tokens`, `parameters`, `FLOPs/token` or `bits/byte`. Thresholds use the extracted value's unit; the engine does not convert milliseconds to seconds or bytes to MiB. |
| `metrics.<metric>.description` | Optional string | Definition, scope and limitations of the quantity. Descriptive text is not a replacement extractor or a computed formula. |
| `metrics.<metric>.deterministic` | Optional boolean; absent treated as false for validation | Declares that the quantity is fixed by the candidate source/declared configuration rather than run-to-run variation. Required to be true for reference exact assertions and equality constraints. It is a claim about the quantity, not proof of determinism or a request to enable deterministic GPU execution. |
| `metrics.<metric>.never_use` | Optional string | Annotation naming a rounded or otherwise unsuitable alternative field. The extractor does not inspect or blacklist that field automatically; the executable choice remains `extract`. Unused in current tasks. |
| `metrics.<metric>.extract` | Required object | One of the extractor configurations below. All nine tasks currently use `metrics_json` for every declared metric. |
| `metrics.<metric>.extract.kind` | Required string | Selects `metrics_json`, `summary_field`, `block_field`, `source_constant`, `source_regex` or `product`. All six are implemented in [metric extraction](../engine/evaluation/metrics.py); the latter five are not used by the nine current definitions. |
| `metrics.<metric>.cross_check` | Optional object; absent means no cross-check | Consistency check against another declared metric. Unused in current tasks. It does not substitute a second measurement when one is missing. |
| `metrics.<metric>.cross_check.against` | Required string when `cross_check` exists | Name of another declared metric. |
| `metrics.<metric>.cross_check.scale` | Required number when `cross_check` exists | Multiplier converting the other value into this metric's units. |
| `metrics.<metric>.cross_check.tolerance` | Required number when `cross_check` exists | Absolute allowed difference in this metric's units: `abs(value - other * scale) <= tolerance`. No implicit relative tolerance. |

Cross-checks run after extraction. If either value is `null`, the check is
skipped. A disagreement sets the metric carrying `cross_check` to `null` and
records an extraction error; the other metric's extracted value remains in
the result. Cross-check tolerance is not an objective tie band.

#### Extractor configuration fields

| Field | Type / required condition | Meaning |
| --- | --- | --- |
| `metrics.<metric>.extract.field` | String; required for `metrics_json`, `summary_field`, `block_field` | Literal output key or text label. For `metrics_json` it is a direct object key, not a dotted JSON path. |
| `metrics.<metric>.extract.block` | String; required for `block_field` | Text header whose stripped line must equal this value's stripped text. |
| `metrics.<metric>.extract.strip` | Optional array of strings for `summary_field` or `block_field`; default empty | Tokens removed by string replacement before whitespace trimming and coercion. This is not a regular expression or a set of characters to trim only at the ends. |
| `metrics.<metric>.extract.file` | String; required for `source_constant` or `source_regex` | File relative to this launch's candidate source root. Reading the pristine reference instead would describe the wrong program. |
| `metrics.<metric>.extract.name` | String; required for `source_constant` | Name of a module-level Python assignment to read without executing candidate code. |
| `metrics.<metric>.extract.pattern` | String; required for `source_regex` | Python regular expression searched over the source file; the result is a boolean. Pattern matching is advisory and may not directly supply an objective or constraint metric. |
| `metrics.<metric>.extract.factors` | Array of at least two strings; required for `product` | Declared metric names whose extracted values are multiplied. Units must be chosen consistently in the declaration; the engine performs no dimensional analysis. |

The extractor kinds have different lookup and failure behavior:

| Kind | Executable behavior |
| --- | --- |
| `metrics_json` | Normalizes carriage returns, finds lines beginning `METRICS_JSON: ` followed by an object, and parses the last matching record. Reads `field` at recorded JSON precision. Missing keys or JSON `null` yield `null`; a missing or malformed canonical record produces extraction errors. There is no fallback to a human-readable summary. |
| `summary_field` | Reads the last matching `field: value` line beginning at column zero. Missing label yields `null`. Optional `strip` tokens are removed before coercion. |
| `block_field` | Finds the first matching block header, then the first matching `field: value` before the next blank line; indentation is allowed. Missing block/field yields `null`. It does not use the last block. |
| `source_constant` | Parses module-level assignments with Python's AST and folds numeric constants, earlier names, unary signs and supported arithmetic (`+`, `-`, `*`, `/`, `//`, `%`, bounded `**`). Later assignments to the requested name take precedence. Missing name yields `null`; missing source, invalid syntax or an unresolvable assignment to that name produces an extraction error. Function calls are not executed. |
| `source_regex` | Uses a regular-expression search over the candidate file, returning whether a match exists. Missing source produces an extraction error. This is a source heuristic, not an instrumented behavioral measurement. |
| `product` | Resolves dependencies, multiplies the factors, and coerces the result to the declared type. Any `null` factor gives `null`; unresolved dependency cycles produce extraction errors. Nested products can resolve when their dependencies do. |

`deterministic: true` does not mean “read from source”: parameter count and
counted FLOPs are instrument outputs declared deterministic. Conversely, a
printed timer is still variable. Parsing stdout also does not authenticate its
author. Current tasks require the immutable reporter to emit the only official
record and prohibit candidate-authored replacement `METRICS_JSON` lines.

### Metrics used by the nine tasks

These are all 16 metric keys declared across the nine definitions. Each uses
`extract.kind: "metrics_json"` and an `extract.field` equal to its key. The
determinism column records the JSON declaration, not an independent empirical
test. Extra quantities printed by `prepare.py` are not automatically declared
metrics or additional objectives.

| Metric path | Type; unit; deterministic | Meaning and tasks declaring it |
| --- | --- | --- |
| `metrics.val_bpb` | number; bits/byte; false | Final validation bits per byte on the fixed held-out data, measured by `evaluate_bpb` in the reporter. Quality gate in all nine tasks. |
| `metrics.num_params_total` | integer; parameters; true | Exact count from the model's parameters, with unique tensors as returned by `model.parameters()`. All nine tasks. |
| `metrics.peak_vram_bytes` | integer; bytes; false | Peak PyTorch-allocated CUDA memory from setup through the reporting read, including FLOPs probing, training and validation at the frozen `128 × 2048` evaluation shape. All nine tasks. In serving tasks this is read before resetting the counter and running inference probes. It excludes reserved-but-unused allocator memory and allocations outside PyTorch's accounting. |
| `metrics.training_data_tokens_available` | integer; tokens; true | Configured token-position allowance exposed by the fixed shuffled training loader, pinned to `631241817` in all nine tasks. It is not the number of unique positions actually consumed; repeated use is allowed and counted separately. |
| `metrics.flops_per_token_measured` | integer; FLOPs/token; true | Counted forward/backward operations on the uncompiled model using registered formulas and the attention tally, divided by probe tokens. Includes counted matrix, convolution and attention work; excludes optimizer updates and unregistered operations, including elementwise/normalization work. All nine tasks. It is not a hardware counter or the analytic FLOPs estimate. |
| `metrics.total_tokens` | integer; tokens; false | Running sum of training token uses passed to the reporter, including accumulation, warmup and repetitions. Equals completed updates times tokens per update when update size is constant. Declared by `flops`, `memory`, `params`, `traintime`. |
| `metrics.train_seconds_to_target` | number; s; false | Harness training seconds at the first scheduled full evaluation below the target, excluding evaluation time and the harness's ten warmup updates. `null` when no passing probe occurs. Declared by `traintime`. |
| `metrics.updates_to_target` | integer; optimizer steps; false | Completed logical optimizer updates, including warmup, at the first passing scheduled probe. `null` without a passing probe. Declared by `data`, `steps`. |
| `metrics.tokens_to_target` | integer; tokens; false | Summed token uses, including warmup and repetitions, when the first scheduled probe passes. `null` without a passing probe. Declared by `data`, `steps`. |
| `metrics.target_val_bpb` | number; bits/byte; true | Target supplied to and reported by the harness. Declared by `data`, `steps`, `traintime` and constrained to equal `1.05`. This is a setting reported back, not measured validation quality. |
| `metrics.request_ms_median` | number; ms; false | Median of 30 synchronized, teacher-forced requests after 3 warmup passes: 1,536-token prefill plus 512 single-token steps. Prefill is timed; state allocation and reset occur outside the timer. Declared by `decode`, `kvcache`, `request`. |
| `metrics.nopref_request_ms_median` | number; ms; false | Same timing protocol with one prefill token and 512 decode steps. Declared by `decode`, `request`. |
| `metrics.kv_cache_bytes` | integer; bytes; false | Allocation growth while holding one `graph=False` state after prefill and all 512 decode steps, following a discarded warmup. Uses prefill 1,536 and `max_len=2048`; includes approximately 1 MiB of allocator rounding. Declared by `kvcache`, `request`. Measures retained device state, not transient peak memory. |
| `metrics.nopref_kv_cache_bytes` | integer; bytes; false | Same retained-allocation measurement with prefill 1 and `max_len=513`. Declared by `decode`, `kvcache`. |
| `metrics.decode_tv_distance_max` | number; probability; false | Maximum total-variation distance, `0.5 * sum(abs(p_decode - p_forward))`, across 513 scored next-token distributions for the 1,536-prefill request. Compared with the same model's forward predictions using teacher-forced inputs. Declared by `decode`, `kvcache`, `request`. |
| `metrics.nopref_decode_tv_distance_max` | number; probability; false | Same fidelity statistic for the one-token-prefill request. Declared by `decode`, `kvcache`, `request`. |

Here `ms` means milliseconds, `s` seconds, and MiB in explanatory text means
`2^20` bytes. A logical optimizer step includes its accumulation microbatches.
To-target measurements locate the first observed passing probe, not an
interpolated threshold crossing; probes become due every 30 harness training
seconds and run at update boundaries.

### Objective, eligibility and comparison

All nine current definitions use the following explicit form. A metric name
must already be declared in `metrics`; a source-regex heuristic cannot directly
decide an objective or constraint. The lists are required even when empty.

| Field | Type / presence | Meaning |
| --- | --- | --- |
| `objective.comparison_mode` | Required string, exactly `"gated"` | Apply constraints and a quality threshold before comparing the optimization target. |
| `objective.target` | Required object | Optimization target, containing exactly `metric` and `direction`. |
| `objective.target.metric` | Required string | Metric to optimize after eligibility checks. See the overview and task descriptions for each task's target. |
| `objective.target.direction` | Required string: `minimize` or `maximize` | Which direction improves the target. Every current task minimizes. |
| `objective.quality_gate` | Required object | Quality threshold, containing exactly `metric`, `operator` and `value`. |
| `objective.quality_gate.metric` | Required string | Metric tested for quality eligibility; `val_bpb` in all nine tasks. |
| `objective.quality_gate.operator` | Required string: `<`, `<=`, `>`, `>=` or `==` | Exact numerical comparison. All nine use `<`, so a value of exactly `1.05` fails their quality gate. |
| `objective.quality_gate.value` | Required number | Threshold in the gate metric's units; `1.05` bits/byte in all nine tasks. |
| `objective.tiebreaks` | Required array of objects | Ordered metrics consulted when target values are exactly equal. Each entry contains exactly `metric` and `direction`; `[]` means no declared tiebreaks. |
| `objective.tiebreaks[].metric` | Required string | Declared metric for that tiebreak. |
| `objective.tiebreaks[].direction` | Required string: `minimize` or `maximize` | Preferred direction for that tiebreak. |
| `objective.constraints` | Required array of objects | Numeric eligibility clauses, evaluated in listed order. Each contains exactly `metric`, `operator`, `value` and `basis`. `[]` means no additional numeric constraints. |
| `objective.constraints[].metric` | Required string | Declared quantity bounded or pinned. It may not be the gate metric or the optimization target. |
| `objective.constraints[].operator` | Required string: `<`, `<=`, `>`, `>=` or `==` | Comparison against the bound. `==` is allowed only for a metric declared deterministic; it pins the quantity without tolerance. |
| `objective.constraints[].value` | Required number | Bound in that metric's native units, for example bytes rather than MiB for `peak_vram_bytes`. |
| `objective.constraints[].basis` | Required nonempty string | Explanation of how the bound was set. Appears in breach reasons. It neither estimates a new bound nor relaxes the recorded number. |

The [objective implementation](../engine/evaluation/objective.py) first checks numeric
constraints, then the quality gate, then the validity of present ranking
values. A missing constraint or gate value fails eligibility. Every present
target **and tiebreak** must be strictly positive and finite. Missing ranking
values are handled by comparison/ranking rather than that validity check.
These validity rules are implemented globally; there is no `minimum_target`
field in the task JSON.

For pairwise gated comparison, an eligible candidate wins when its target
strictly improves. On an exact target tie, tiebreaks are consulted in order;
unmeasured tiebreaks are skipped. Equal values throughout are not an
improvement. Quality need not improve further once its gate is satisfied.
There is no rounding, statistical tolerance or minimum effect size in this
comparison.

Collection uses `rank_key`, which requires **all** declared ranking metrics to
be measured. It sorts by target and tiebreaks in order, then by larger quality
gate margin if all those values tie. For these tasks the margin is
`1.05 - val_bpb`. This last sorting preference is implemented in the engine,
not declared as a further tiebreak in JSON; it does not make a pairwise tie
an accepted improvement. Thus pairwise acceptance and report ordering are
related but not identical. A method's internal selection policy remains its
own.

### Reference launch

The reference runs the unmodified packaged source before candidates. Current
definitions contain assertions but no embedded reference measurements. Actual
measurements come from this run's reference result and are not replaced by
numbers quoted in descriptions or constraint bases.

| Field | Type / presence | Meaning |
| --- | --- | --- |
| `reference_launch.position` | Required integer, exactly `1` | Contractual initial position of the reference. The lifecycle measures it before method candidates; collection also reports its actual ledger sequence. |
| `reference_launch.assert_exact` | Required nonempty object | Map of deterministic metric names to expected reference values. Each key must name a declared metric with `deterministic: true`. |
| `reference_launch.assert_exact.<metric>` | Number or boolean | Exact expected value in that metric's units. Missing or unequal extracted values are reproduction failures. These assertions apply to the reference, not to every candidate. Candidate bounds belong in `objective.constraints`. |
| `reference_launch.assert_exact.num_params_total` | Integer where present | `50332176` in `data`, `decode`, `flops`, `kvcache`, `request`, `steps`, `traintime`. Not asserted by `memory` or `params`. |
| `reference_launch.assert_exact.flops_per_token_measured` | Integer where present | `239078400` FLOPs/token in `decode`, `kvcache`, `memory`, `params`, `request`. Not asserted by `data`, `flops`, `steps`, `traintime`. |
| `reference_launch.assert_exact.training_data_tokens_available` | Integer in every task | `631241817` available training token positions. |

[Reference verification](../engine/evaluation/reference.py) compares exact assertions
numerically without a tolerance. The current lifecycle additionally requires
a resolved reference with status `ok` before starting the method. Exact
reproduction and objective eligibility are separate checks: passing the
assertions alone does not establish that measured quality passed its gate.
Reference eligibility is calculated from the measured result using the same
quality gate and constraints as candidates.

### Reporting

| Field | Type / presence | Meaning |
| --- | --- | --- |
| `report.headline` | Required string | Declared metric used as the report's headline and trajectory value. All nine tasks choose their optimization target. This field alone does not choose the best candidate. |
| `report.headline_direction` | Optional string: `minimize` or `maximize`; default `minimize` | Direction used for headline display and improvement calculation. All nine explicitly use `minimize`, matching their target. |
| `report.columns` | Required nonempty array of strings | Ordered declared metric names for the displayed metric projection. Must include `headline`. All current tasks include the metrics needed to explain their objective and constraints. |

[Collection](../engine/records/collect.py) retains both the selected columns and the
full extracted metric dictionary. It selects the best result using the task's
objective rank key, including eligibility, rather than taking a minimum of
the headline column or trusting the method's own best-result record.
Omitting a metric from `columns` would not remove its objective constraint.
[Reports](../engine/submission/comparison.py) and [submission summaries](../engine/submission/data.py)
consume these collected values.

### Executable source contract

The JSON identifies the instrument and its rules; the neighboring source
implements the measurement. All packages supply `train.py`, `prepare.py`,
`pyproject.toml` and `uv.lock`. Use the complete package as candidate source,
prepare its data and dependencies, and submit candidate execution through the
[experiment interface](../engine/API.md). Task loading does not fetch upstream
code or prepare data on its own.

In the training source, the required path is:

1. Build the model, construct training data through `prepare.make_dataloader`,
   and print the GPU-work witness after allocation.
2. Run `prepare.measure_flops_dispatch` once on the actual model and first
   training microbatch. The instrument selects up to eight complete sequences
   and measures uncompiled forward/backward work. Pass that return value
   unchanged to the reporter.
3. Train under `TIME_BUDGET`. Where the task supplies a target, construct
   `TimeToTargetHarness` with the prescribed target and frozen defaults,
   call `tick` once per completed logical update with its synchronized duration
   and actual token count, and stop when it reports the first pass.
4. Call `prepare.report_efficiency_metrics` once with the actual trained model,
   tokenizer, counts and durations. Supply the harness used by this run for the
   three to-target tasks; the other tasks declare `harness=None`. The reporter
   remeasures final validation quality, counts parameters, reads the allocation
   peak and emits the canonical JSON record.

The [reference loop](traintime/code/train.py) checks `TIME_BUDGET = 600` at
completed update boundaries, so its final update can cross the limit.
Its training clock starts with update 12; the
[to-target harness clock](traintime/code/prepare.py) starts with update 11.
Both exclude validation time. The harness schedules full quality probes every
30 training seconds and stops the loop on the first passing probe.

The reporter directly measures some quantities and records others supplied by
the training loop. For example, it reads validation quality and parameter count
itself, but records `total_tokens` and the passed FLOPs-probe return value.
The harness sums the durations and token counts given to `tick`. Immutable
reporter bytes therefore do not prove that arguments describe the actual
work; the complete `frozen_regions[].reason` and `known_constraints` texts
define that obligation.

For the serving packages, the model must additionally expose the following
methods, checked by the frozen [serving instrument](request/code/prepare.py):

| Method | Required behavior |
| --- | --- |
| `init_decode_state(batch, max_len, graph=True) -> state` | Construct the state owning the request's KV-cache data. Support the requested capacity and the eager `graph=False` measurement path. |
| `reset_decode_state(state) -> state` | Start a fresh request without retaining prior computed request information. Timed passes reuse an allocated state and reset it outside the timer. |
| `decode_step(idx, state) -> (logits, state)` | Process the supplied prefill or single-token input and return next-token logits for the last position. The probe indexes `logits[:, -1, :]`, so preserve that batch/position/vocabulary interface. Use the same parameters as `forward`; eager and captured execution must preserve computation and retained data. |

For `decode`, `kvcache` and `request`, the state object returned by
`init_decode_state` must own all KV-cache data, including compressed
representations or retained history used in its place, so that dropping the
object frees the data. All such data must remain on the GPU throughout the request.

The serving reporter refuses a missing decode protocol before running probes.
Both prefill shapes are measured even if a task declares only some resulting
keys. Teacher-forced decode distributions are compared with `forward` at 513
positions; the maximum total-variation constraints apply to each shape.
The latency probe allocates with `max_len = prefill + steps + 1`
(`2049` or `514`), whereas the retained-state memory probe uses
`prefill + steps` (`2048` or `513`). These are the actual instrument arguments;
they are not additional tunable JSON fields.

## Task descriptions

Each task below describes its objective, constraints and measurement procedure.
The linked `task.json` defines the task contract; its `code/` implements the instrument.

<details>
<summary>data</summary>

# Training tokens

Total training-token uses at the first passing scheduled full validation evaluation,
including warmup updates and repetitions, summed as the run consumed them. This is
not the number of unique dataset positions seen. The configured data pool is fixed
separately. No passing probe during the launch means no qualifying target value.

## Objective and constraints

Minimize `tokens_to_target` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `tokens_to_target` (minimize), `updates_to_target` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `flops_per_token_measured` | `<= 239078400` |
| `num_params_total` | `<= 50332176` |
| `peak_vram_bytes` | `<= 47198976512` |
| `target_val_bpb` | `== 1.05` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

Quality probes occur every 30 training seconds at update boundaries. The result is
the first observed pass, not the exact threshold crossing. Training stops at the first
pass or at `TIME_BUDGET`, the same 600-second budget every task has.

## Files

- [task.json](data/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](data/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>

<details>
<summary>flops</summary>

# FLOPs per token

Forward/backward FLOPs per token counted from executed operations using registered
shape-based formulas for matrix multiplication, convolution and attention. The probe
runs uncompiled; optimizer updates and unregistered operations, including elementwise
arithmetic, are excluded. This is not a hardware-counter measurement.

## Objective and constraints

Minimize `flops_per_token_measured` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `flops_per_token_measured` (minimize), `total_tokens` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `num_params_total` | `<= 50332176` |
| `peak_vram_bytes` | `<= 47198976512` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

## Files

- [task.json](flops/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](flops/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>

<details>
<summary>memory</summary>

# Peak allocated memory

Peak memory tracked by the device's PyTorch allocator through the reporting read,
including setup, the uncompiled FLOPs probe, training and validation. This is not
isolated compiled-training memory or total device memory from all allocation sources.

## Objective and constraints

Minimize `peak_vram_bytes` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `peak_vram_bytes` (minimize), `total_tokens` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `flops_per_token_measured` | `<= 239078400` |
| `num_params_total` | `<= 50332176` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

## Files

- [task.json](memory/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](memory/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>

<details>
<summary>params</summary>

# Model parameters

Exact total parameter count of the model, counted by the immutable instrument.

## Objective and constraints

Minimize `num_params_total` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `num_params_total` (minimize), `total_tokens` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `flops_per_token_measured` | `<= 239078400` |
| `peak_vram_bytes` | `<= 47198976512` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

## Files

- [task.json](params/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](params/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>

<details>
<summary>steps</summary>

# Training steps

Completed logical optimizer updates, including warmup, at the first passing
scheduled full validation evaluation. Each update includes its accumulation
microbatches. `tokens_to_target` records the corresponding token uses and breaks
ties. No passing probe during the launch means no qualifying target value.

## Objective and constraints

Minimize `updates_to_target` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `updates_to_target` (minimize), `tokens_to_target` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `flops_per_token_measured` | `<= 239078400` |
| `num_params_total` | `<= 50332176` |
| `peak_vram_bytes` | `<= 47198976512` |
| `target_val_bpb` | `== 1.05` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

Quality probes occur every 30 training seconds at update boundaries. The result is
the first observed pass, not the exact threshold crossing. Training stops at the first
pass or at `TIME_BUDGET`, the same 600-second budget every task has.

## Files

- [task.json](steps/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](steps/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>

<details>
<summary>traintime</summary>

# Training time

Training seconds at the first passing scheduled full validation evaluation,
measured by the frozen time-to-target harness in `prepare.py`. Evaluation time
and the first ten optimizer updates are excluded from this clock. No passing
probe during the launch means no qualifying target value.

## Objective and constraints

Minimize `train_seconds_to_target` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `train_seconds_to_target` (minimize), `total_tokens` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `flops_per_token_measured` | `<= 239078400` |
| `num_params_total` | `<= 50332176` |
| `peak_vram_bytes` | `<= 47198976512` |
| `target_val_bpb` | `== 1.05` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

Quality probes occur every 30 training seconds at update boundaries. The result is
the first observed pass, not the exact threshold crossing. Training stops at the first
pass or at `TIME_BUDGET`, the same 600-second budget every task has.

## Files

- [task.json](traintime/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](traintime/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>

<details>
<summary>decode</summary>

# Decode latency

Median elapsed time over 30 timed requests after 3 warmup passes, with one prefill
token and 512 decode steps. Inputs are teacher-forced through the candidate's
`decode_step`, with synchronized GPU completion inside the timer. State allocation
and reset are outside the timer. Context grows from 1 to 513 tokens; this uses the
same timing protocol as `request` with a shorter initial context.

## Objective and constraints

Minimize `nopref_request_ms_median` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `nopref_request_ms_median` (minimize), `request_ms_median` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `nopref_kv_cache_bytes` | `<= 10485760` |
| `decode_tv_distance_max` | `<= 0.05` |
| `nopref_decode_tv_distance_max` | `<= 0.05` |
| `flops_per_token_measured` | `<= 239078400` |
| `num_params_total` | `<= 50332176` |
| `peak_vram_bytes` | `<= 47198976512` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

## Files

- [task.json](decode/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](decode/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>

<details>
<summary>kvcache</summary>

# GPU-resident KV-cache memory

PyTorch-allocated GPU bytes retained by one live request state. After a discarded
warmup request, the instrument reads `memory_allocated` before constructing the
state and after prefilling 1,536 tokens and running all 512 decode steps, dropping
temporary logits. The difference is measured at `max_len=2048` with `graph=False`
to exclude CUDA graph pools. It measures retained allocation rather than transient
peak memory and assumes no particular state structure. The reading includes
approximately 1 MiB of allocator block rounding.

## Objective and constraints

Minimize `kv_cache_bytes` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `kv_cache_bytes` (minimize), `nopref_kv_cache_bytes` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `request_ms_median` | `<= 750` |
| `decode_tv_distance_max` | `<= 0.05` |
| `nopref_decode_tv_distance_max` | `<= 0.05` |
| `flops_per_token_measured` | `<= 239078400` |
| `num_params_total` | `<= 50332176` |
| `peak_vram_bytes` | `<= 47198976512` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

## Files

- [task.json](kvcache/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](kvcache/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>

<details>
<summary>request</summary>

# Request latency

Median elapsed time over 30 timed requests after 3 warmup passes, with a
1,536-token prefill and 512 decode steps. Inputs are teacher-forced through the
candidate's `decode_step`. Prefill and synchronized GPU completion are inside
the timer; state allocation and reset are outside it. Context grows from 1,536
to 2,048 tokens.

## Objective and constraints

Minimize `request_ms_median` among candidates satisfying the quality gate
`val_bpb < 1.05` and every constraint below.
The task's ranking order is `request_ms_median` (minimize), `nopref_request_ms_median` (minimize).
These rules determine official eligibility and ranking, not how a method must search.

| Quantity | Required condition |
| --- | --- |
| `kv_cache_bytes` | `<= 41943040` |
| `decode_tv_distance_max` | `<= 0.05` |
| `nopref_decode_tv_distance_max` | `<= 0.05` |
| `flops_per_token_measured` | `<= 239078400` |
| `num_params_total` | `<= 50332176` |
| `peak_vram_bytes` | `<= 47198976512` |
| `training_data_tokens_available` | `== 631241817` |

The target and tiebreak must be positive and finite. The budget is 100
launches including the reference, with a 600-second training-clock limit,
1 GPU per launch, reference seed 42. GPU-work failures count;
no-work failures are classified by the shared accounting rule.

## Files

- [task.json](request/task.json): exact machine-readable definition, including hardware,
  measurement protocol, metrics, reference assertions and all constraints.
- [code/](request/code/): runnable reference, with `train.py` editable.
  `prepare.py`, `pyproject.toml`, `uv.lock` remain immutable.

Copy the complete `code/` directory to build a candidate. Run it only through the
benchmark evaluator under the task's declared environment and limits.

</details>
