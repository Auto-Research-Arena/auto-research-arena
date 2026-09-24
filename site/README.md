# Public leaderboard

This directory contains the static website and its submission data. Every run
uses the same `autoarena/site-submission/1` JSON format: method and target IDs,
winner and reference values, budget usage, research-model and compute metadata,
and the recorded measurements. The website derives its charts from those
measurements and compares the supplied results.

## Add a run

Export the run with the engine, then convert its results for the website:

```bash
python3 -m pip install -r site/requirements.txt
python3 scripts/leaderboard.py add \
  --submission runs/my-run/submission/submission.json \
  --output site/submissions/my-method/params
```

The command creates two files:

```text
site/submissions/my-method/params/
  submission.json
  artifacts.json
```

`submission.json` contains the website result record. `artifacts.json` contains
download links when available. It can be empty:

```json
{}
```

The page shows only the links supplied. To include one archive with code and logs,
pass `--artifacts-url HTTPS_DOWNLOAD_URL` to `add`. Alternatively, use
`--code-url URL` and `--logs-url URL` separately. `--report-url URL` is optional.
Links use HTTPS without credentials or expiring query parameters.

Repeat for other targets or runs, giving each run its own folder. A submission
PR may contain all targets for a method. Include its public name and a source
and license link when available. Commit the JSON records and artifact links;
large code and log archives stay outside Git.

See the [root README](../README.md#4-export-a-submission) for instructions on exporting a run.

## Import organized results

Convert the organized publication data into a new directory:

```bash
python3 scripts/import_publication.py \
  --organized runs/publication-20260920/organized \
  --output runs/imported-site-submissions
python3 scripts/leaderboard.py check --directory runs/imported-site-submissions
python3 scripts/leaderboard.py build \
  --directory runs/imported-site-submissions \
  --output runs/leaderboard-site
```

The importer includes every catalog entry, uses the repository's method and
target names, preserves the recorded numbers, and writes empty artifact objects.
The initial website data contains seven methods across nine targets.

## Result records and charts

The website checker verifies JSON structure, unique ordered measurement
sequences, budget counts, and agreement of the winner and reference with the
recorded measurements. Every measurement uses exactly the metric fields declared
in its target's `task.json`, with `null` for unavailable values. Conversion omits
extra instrument diagnostics. Each JSON file has a SHA-256 in the generated display
data. Checks and builds do not contact artifact hosts.

Chart steps count charged launches, including the reference and failures.
Qualifying measurements supply the plotted points and the best-so-far curve.
Each curve ends at the run's recorded budget usage. The individual-experiment
toggle shows the qualifying points, including results worse than the incumbent.
The downloadable run JSON contains every recorded measurement.

The overall table has one row per method and one column per target. If a method
has several runs on a target, its best recorded ranking key supplies that cell.
Exact ranking ties share a rank; all runs remain available in target details.
Selecting a result opens that target and isolates the selected run in the chart.

## Method names and source links

[methods.json](methods.json) records display names and optional source and
license links, keyed by the IDs used in `examples/`:

`arbor`, `autoscientists`, `beam-search`, `gear`,
`heuresis-map-elites`, `sequential-search`, and `tpe`.

An unlisted method displays its submitted ID. **Open source** shows **Yes** when
both a public source and license are recorded, linking to the source; otherwise
it shows **No**. These display fields do not affect result ranking.

## Overall Elo

Overall Elo uses the margin-aware calculation from the AutoArena results page:

- Compare each pair of methods on shared targets, using their best result per
  target. Lower values win, equal values tie, and missing targets add no comparison.
- Give every included target equal weight (`1/9` for all nine targets).
- Round training time with `30 * floor(raw_seconds / 30 + 0.5)` before
  ranking and display. Matching checkpoints tie. Tables, points and progress
  curves use the rounded values; downloaded submissions retain raw measurements.
  The selected candidate stays unchanged.
- Normalize each absolute log ratio by the target's median nonzero pairwise log
  gap, floored at `1e-9` (or 1 when every result ties). The margin multiplier is
  `min(3, log1p(normalized_gap) / log(2))`; ties have zero update.
- Start at 1500. With the standard 400-point logistic expectation, update by
  `32 × (target_weight / average_target_weight) × margin × (outcome − expectation)`.
- Average 4,000 shuffled comparison orders using NumPy's default generator and
  seed `20260902`. Equal and opposite updates preserve the cohort mean of 1500.
  The leader has no fixed rating or maximum.

For reproducibility, [elo.json](elo.json) specifies the method order
(Sequential Search, Beam Search, AutoScientists, Arbor, GEAR, Heuresis, TPE),
target order (parameters, FLOPs, memory, training time, steps, data, KV cache,
decode, request), and rounding steps. Additional methods follow in sorted ID
order. Within each target, compare every unordered pair in that method order.
Only the selected endpoint values enter Elo; reference scores, progress curves
and trial counts do not affect it.

The full-precision ratings, comparison count and calculation settings appear in
`ranking.json` and `leaderboard.json`. `match_order_sd` is the population standard
deviation across shuffled passes: it describes order sensitivity, not
experimental uncertainty. The page displays Elo to one decimal place.
Disconnected comparison groups or nonpositive target values have no overall
rating; their per-target results remain available.

## Build and preview

```bash
python3 scripts/leaderboard.py check
python3 scripts/leaderboard.py build --output runs/leaderboard-site
python3 -m http.server 8000 --directory runs/leaderboard-site
```

Open `http://localhost:8000`. The build writes HTML, CSS, JavaScript,
`leaderboard.json`, `ranking.json`, and downloadable JSON records under `submissions/`.
It needs no frontend package manager or external chart service.

Publish the generated directory with your static web host. Assets use relative
paths, including when the site is hosted beneath a project path.
