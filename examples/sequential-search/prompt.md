# Sequential Search

Read `AUTOARENA_CONTEXT`, `source/settings.json`, the policy files in
`source/policy/` and the complete `task_dir/task.json`, including its constraints.
Start with `task_dir/code/` and this run's already measured reference.

Replace experiment execution with `autoarena evaluate --source ABS_CANDIDATE
--request-id ID --research-log-file ABS_LOG_JSON`. Before each measurement,
write the card into a nonempty `ideas` list and the current `status` in that JSON.
`ABS_LOG_JSON` is the absolute path to this per-measurement JSON file (for example,
`research-log.json`), separate from the method's `search.jsonl`.
Give the returned canonical result and logs to the native evaluator. Preserve
request IDs when recovering interrupted calls and never remeasure the reference.

Run full proposer, executor and evaluator rounds, each including incumbent
publication, until the task's launch budget is spent, then call `autoarena finish`
with the actual stopping reason. There is no round count to reach: the budget in
the frozen task is the only limit, and charged launches include the reference and
every failure after GPU work starts, so read the spend from the benchmark rather
than counting your own rounds.

Maintain one incumbent and evaluate one child per round with one measurement
worker. Follow the supplied policy files and settings.

Keep a persistent driver, but create fresh proposer, executor and evaluator coding-agent sessions each round with explicit task, parent source, history from `search.jsonl` and `PROGRESS.md`, and policy. The driver must not silently replace these research roles.

The parent source is the saved code identified by `search_state.json.incumbent_source`.

A composite card may coordinate edits serving one attributable mechanism, including necessary constants, but not bundle unrelated ideas. Re-read actual parent source each round and derive available mechanisms; the constants block is not the search boundary and constants remain legal.

Preregister each card with its internal UTC timestamp and current position in the benchmark's `launches.jsonl` ledger before measurement. Predict exact source-derivable quantities with arithmetic. Never predict quality, time/updates/tokens to target, completed training steps or total trained tokens, including numeric deltas, ranges or bounds; for non-derivable targets explain derivable factors instead.

Build isolated complete parent copies, check actual edits and deterministic predictions on CPU, evaluate once, and give the fresh evaluator the complete canonical result and source evidence. Publish the decision and actual incumbent source before the next proposer starts, recording the decision in `search.jsonl` and the incumbent in `search_state.json`.

Keep four objective-neutral retune attempts per structural incumbent. Neutrality is exact measured equality on the ranking key (the metric named by `objective.target.metric`), not the card label. Each neutral attempt consumes a slot regardless of acceptance. Reset only after an accepted strict improvement of the ranking key.

For an eligible candidate, accept a strict improvement of the target. On an exact target tie, while retune allowance remains, accept if either the task's official tiebreak improves or the measured quality-gate margin strictly improves. Otherwise reject. Every exact-target tie consumes a retune attempt, regardless of acceptance. No tolerance bands, epsilon improvement or invented quality prediction.

Seek worthwhile candidates within the task cap; do not pad with duplicate code. Finish on cap after feedback. Resume the same incumbent and retune count from `search_state.json`, with cards and history from `search.jsonl` and `PROGRESS.md`, completing outstanding feedback before new research.

## Research roles

Give each role the task, settings, active policies and relevant research history
from `search.jsonl` and `PROGRESS.md`, together with the inputs below. Save their
outputs in the supplied workspace.

1. **Propose.** Give the proposer the current parent source and metrics, its
   measured gate margin, the remaining retune allowance and launch budget. It
   proposes exactly one idea at the declared scope. Its motivation must explain
   why this is the next attempt, citing an observation, limitation or prior result.

2. **Execute and measure.** Give the executor the card and parent source. It
   implements the idea and submits the candidate through the benchmark API,
   returning the candidate source, canonical result and logs.

3. **Evaluate.** Give the fresh evaluator the card, parent and candidate sources,
   their metrics, the canonical result and logs, and the current retune balance.
   It reports the metrics, the accept/reject decision and what was learned.
   Reject failed or inadmissible candidates before comparing objectives; otherwise
   apply the task's ranking and the method's retention rule.

4. **Publish and continue.** On acceptance, the driver makes the measured
   candidate the new incumbent; on rejection, it keeps the parent. Update the
   retune balance from the measured result and save the decision, feedback and
   incumbent before the next round.

## Records and continuation

The driver maintains three files in the supplied workspace:

- **`search_state.json`** — `incumbent_id`, `incumbent_source` (a saved source
  path or hash), `incumbent_metrics` and `retunes_spent_on_incumbent`. The measured
  reference is the first incumbent. Rewrite the state atomically after each round
  and keep its source available.
- **`search.jsonl`** — one compact transition per round: parent, candidate,
  proposal, motivation, measured metrics, decision, compared values and deciding
  rule, gate margins, `objective_neutral` and `retunes_spent_on_incumbent`
  after the round.
- **`PROGRESS.md`** — a journal entry when an idea is selected, before execution,
  and another after evaluation with the decision and findings.

Keep `search.jsonl` and `PROGRESS.md` append-only; append corrections rather than
rewriting earlier entries. The records must support resume without chat history.

## Method settings

```json
{
  "idea_scope": "composite",
  "retune_slots_per_incumbent": 4
}
```
