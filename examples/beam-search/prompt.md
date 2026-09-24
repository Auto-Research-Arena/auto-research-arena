# Beam Search

The driver owns the research loop and frontier; the engine measures candidates.

Read `source/settings.json` using the `source` directory in `AUTOARENA_CONTEXT`,
the policy files under `source/policy/`, and the complete `task_dir/task.json`,
including its constraints. Start with `task_dir/code/` and this run's already
measured reference. The task JSON controls measurement and ranking.

Run whole generations, including feedback and frontier publication, until the
task's launch budget is spent. Then call `autoarena finish` with the actual completed
generation count. There is no generation count to reach: the budget in the frozen task
is the only limit, and charged launches include the reference and every failure after
GPU work starts, so read the spend from the benchmark rather than counting your own
generations. Preserve earlier recorded stopping reasons. Keep the complete-generation
boundary: if the remaining budget cannot fit a whole generation, record the unused
budget and stopping reason rather than shrinking the beam or padding it. A worker wave is not a
generation.

Begin with the reference as a width-one frontier and grow to at most four. Freeze the whole frontier per generation, propose two children per member, interleave parents round-robin across four workers, and resolve the whole generation before any frontier publication. One worker wave need not equal one generation.

Create fresh proposer, executor and evaluator coding-agent sessions every generation with explicit frozen parent sources, histories, task and policy. Re-read each actual source and derive mechanism availability. Composite edits must serve one attributable mechanism; necessary constants and constant changes are legal.

Use the coding-agent runtime's tools or commands to create these role sessions.
Do not simulate roles using inline reasoning, prose or textual tool-call markup.
If separate role sessions are unavailable, record an integration failure before
any candidate measurement. Persist actual returned role session IDs,
their input paths, and their outputs in this workspace. Delegate research to those
roles; the persistent driver coordinates their work and complete-generation
selection. All roles see only this task, supplied policies, and this
run's candidates and research history. Never load another run's search artifacts.

Keep parent/task/policy inputs read-only and use isolated candidate directories.
Persist every candidate, preregistered card, and the complete batch request before
submission. Use `autoarena evaluate --request-id ID --batch FILE`, putting
`candidate_id`, `source`, `parent_id`, and `research_log` in every item. The
`research_log` object carries the card in a nonempty `ideas` list, a nonempty
`status` string describing the current research state, generation number and role
provenance. Preserve the exact payload and source bytes on recovery. All candidate
GPU work must use this API. CPU syntax checks do
not constitute measurements. Do not run training directly.

Preregister cards with UTC timestamps and the benchmark's `launches.jsonl` ledger position before measurements. Predict only exact source-derivable quantities with arithmetic. No predictions of quality, time/updates/tokens to target, completed training steps or total trained tokens, including numeric deltas, ranges or bounds.

Use two retune slots per generation and a two-generation stall window.
Decide lane availability before proposals: it opens only if the best measured
ranking key (the metric named by `objective.target.metric`) strictly improved in
the preceding two completed generations. Record
the exact historical values used. With no qualifying prior improvement, allocate
zero retune slots; do not fabricate bootstrapping progress. Classify neutrality by exact measured ranking-key equality with the parent, not the proposed label. If unexpected neutrality exceeds the allowance, retain the real measurements and report the unresolved policy violation; do not retrospectively invent an allocation.

The installed `autoarena` client supplies raw evaluation results. Use this source's
read-only `selection.py` helper for exact task arithmetic. Create a pool JSON list
whose entries have `id`, `parent_id` where applicable, `source` (the actual immutable
candidate directory), `status` and the full canonical `metrics` object. Run:

```bash
python <source>/selection.py --task <task_dir>/task.json --pool <workspace>/pool.json --output <workspace>/pool-inspection.json
```

Replace placeholders with the context paths. Use a new output filename each time;
the helper refuses overwrites. It computes exact program identity, each eligibility
clause, strictly positive finite ranking values, every task-defined ordered ranking
metric and gate margin. Excluded/unmeasured entries have no usable rank. It does
not choose ideas, collapse retunes or publish the frontier; those remain your
evaluator's work. Do not edit the helper, settings or policy files to fit
an observed result. Current tasks encode all permitted experiment inputs in their
source files; do not invent external task variables.

Pool all frozen parents and successfully measured children. Deduplicate actual task-program identity, which covers the task definition and its declared source files. Remove benchmark-ineligible members before ranking. Only when the retune lane is open this generation, collapse exact-target parent/child pairs to the passing member with larger gate margin. Keep the best four using the task's canonical rank.

Only on an exact remaining rank tie apply the lower-complexity, lower-resource, then older-member fallback without inventing unavailable measurements. Publish one complete generation transition with recorded metrics, source parents, exclusions, retune evidence and the new frontier, then start the next generation.

Never promote a completed subset or fabricate unresolved children. Preserve the complete-generation boundary: if remaining allowance cannot fit two children per frontier member, record the unused budget and stopping reason rather than shrink the beam, select a partial generation or pad.

Finish on cap or when the remaining budget cannot fit a complete generation, after complete feedback. Resume exact unfinished generation membership and request IDs, then complete selection; do not reset retune history, substitute rejected parents or remeasure code.

## One generation

The driver coordinates the fresh proposer, executor and evaluator sessions described
above. Give each role the task, settings, policies and this run's research history,
together with the inputs named below.

1. **Freeze `frontier_before`.** Read the selected frontier from `search_state.json`.
   Every member is a parent for this generation. Keep those parent sources fixed
   until the generation is complete.

2. **Decide whether the retune lane is open, before proposing.** Read the best ranking-key
   value across the frontier for each of the preceding `retune_stall_generations = 2`
   completed generations from `search_state.json`. If it strictly improved at least once
   in that window the lane is open with `retune_slots = 2` slots; if not, this generation
   gets zero. Record which, and the values that decided it.

3. **Ask the proposer for exactly `children_per_parent = 2` children per frontier member**,
   at the declared idea scope. Give it each parent's source and metrics, the research
   history and remaining launch budget. Each motivation must explain why *that particular
   parent* should be expanded in *that* direction. Tell the proposer how many slots are
   open and give it each frontier member's gate margin; it chooses which children are
   intended to be objective-neutral, up to that many, and argues each from mechanism.
   Zero is a valid and usually correct choice. Children beyond the slot count must be
   structural.

4. **Give the executor the cards and their parent sources.** Each child records its
   parent's id and saved source identity, and is prepared in an isolated candidate
   directory based on that exact parent. Check the actual edits and derivable predictions
   on CPU. Return the candidate sources, preregistered cards and complete batch request.

5. **Submit every child through the benchmark API, interleaving parents.** With frontier
   members A/B/C/D each producing two children, the submission order is
   `A1, B1, C1, D1, A2, B2, C2, D2`. The driver submits the saved batch request;
   the API leases whichever of the four workers is available. Complete all candidate
   measurements before selecting the next frontier.

6. **Give the evaluator the cards, parent and child sources, canonical results and logs.**
   Evaluate every completed child, and record what the generation established.
   Classify neutrality from exact measured ranking-key equality with the parent.
   Retain the real measurements and report any retune-allocation overrun.

7. **Form the pool**: `frontier_before` plus all successfully measured children.
   Failed and timed-out children never enter it.

8. **Deduplicate and remove ineligible entries before ranking.** Use `selection.py` as
   described above for program identities, eligibility and canonical ranking values.
   Deduplicate first, on task-program identity. Admissibility ceilings and the gate drop
   in the same step, before any ranking. A dropped entry has no rank and must not remain
   among the ranked values; entries without usable ranking values cannot be selected.

9. **Keep the best `beam_width = 4` entries as `frontier_after`.** Only when the retune lane is
   open this generation, first collapse each parent-child pair sharing an exact ranking-key value
   down to the passing member with the larger gate margin. Rank the survivors using
   the task's canonical rank. Parents remain in the pool, so a child that does not
   improve enough to enter the beam is rejected automatically. Only on an exact remaining
   rank tie apply the lower-complexity, lower-resource, then older-member fallback,
   without inventing unavailable measurements.

10. **Increment the generation and update `search_state.json` atomically.** The driver
    records the complete transition and evaluator feedback in `search.jsonl` and
    `generations.jsonl`, and updates `PROGRESS.md`. Save the selected sources, metrics
    and frontier history needed for the retune window. The selected `frontier_after`
    supplies the parents for the next generation.

## Records and continuation

The driver maintains four files in the supplied workspace:

- **`search_state.json`** — the completed generation, selected frontier with saved
  sources and metrics, and the best frontier ranking values needed for the retune
  window. The measured reference is the initial width-one frontier at generation
  zero. Rewrite the state atomically after each complete generation and keep the
  selected sources available.
- **`search.jsonl`** — one complete transition per generation: the frozen frontier,
  planned children and parent links, request IDs, all outcomes, eligibility
  exclusions, evaluator feedback, measured neutrality and slot counts, and the
  selected frontier.
- **`generations.jsonl`** — the same complete generation transition recorded in
  `search.jsonl`.
- **`PROGRESS.md`** — a journal entry when the generation's ideas are selected,
  before execution, and another after evaluation with the decisions and findings.

Keep `search.jsonl`, `generations.jsonl` and `PROGRESS.md` append-only; append
corrections rather than rewriting earlier entries. These are method-owned records;
never edit the engine's launch ledger. The records must support resume without
chat history.

On restart, load the last complete `search_state.json` and reconcile the unfinished
generation with its saved requests and the engine's launch ledger. Resume saved
roles and pending request IDs as needed to settle the same generation before
creating fresh roles for the next one. If the frontier update was interrupted,
recompute it deterministically from the recorded parent and child evaluations,
using the selection procedure above.

## Method settings

```json
{
  "beam_width": 4,
  "children_per_parent": 2,
  "retune_slots": 2,
  "retune_stall_generations": 2,
  "idea_scope": "composite"
}
```
