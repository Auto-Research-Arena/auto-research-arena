# Policy: the retune lane

**Our rule.** An objective-neutral candidate — one whose ranking key lands at exactly its
parent's value — is a legitimate move when it passes the task constraints, because
it can buy gate margin. Beam Search bounds these attempts within a generation and
keeps the lane open only while the frontier makes progress.

## What "objective-neutral" means, and how it is decided

A candidate is objective-neutral when its **ranking key** — the metric named by
`objective.target.metric` — lands at exactly the value its parent recorded,
byte-exact at full recorded precision.

**It is a claim about the measurement, not about the diff.** "Only changed a constant" is the
wrong test. A structural change often needs coordinated non-structural settlements — a
learning-rate group, an init scale, a dtype cast — to work at all, and under `composite` scope
those legitimately travel in one card. A card that moves the ranking key by any amount is not
objective-neutral no matter how small the edit, and a card that leaves it identical is
objective-neutral no matter how large.

**Classification is measured, never declared.** A card states its intent; the driver
decides from the recorded metric what the candidate actually was, and updates the retune allowance
accordingly. A card that meant to remove parameters and landed at an exact tie is a retune
retroactively. A card that declared itself a retune and moved the metric is not one and spends
nothing. Charging a declaration instead of a measurement lets a mis-sized card leak the
budget.

## Why the lane exists at all

Its purpose is **prospective**, and the brief must say so: an objective-neutral candidate can
improve its parent's quality-gate margin, leaving more room for a later structural change
that might worsen quality.

It cannot rescue anything. A candidate that fails the quality gate or another
task constraint is inadmissible and cannot enter the selected frontier. The lane
fires only on candidates that were admissible in the first place. What survives a rejection is
the *mechanism*, which lives in the card and the evaluator's notes; a later generation may re-apply
it to a parent that has since banked more margin. Buying that margin ahead of the attempt
is the whole point.

## Why it is bounded, and why a human sets the bound

The proposer faces an incentive it cannot see past. An objective-neutral candidate **cannot
worsen the ranking key** — it is a risk-free bid — so a generation-scoped agent maximising its own
generation prefers it to a structural attempt that might breach the gate. Generation after generation of that
is exactly the failure where a run spends its budget tuning numbers, and it arrives through a
sequence of individually defensible decisions.

The ceiling cannot be delegated to the proposer, for a structural reason rather than a
motivational one: roles are fresh subagents per generation, so no proposer remembers what the
previous one chose. **A budget nobody can remember is not a budget.** It is a number in the
config, and the remaining allowance for the generation is passed into every brief.

## Two bounds

[settings.json](../settings.json) sets `retune_slots` to **2** and
`retune_stall_generations` to **2**.

Two bounds, because the two failure modes are different:

- `retune_slots` bounds the **mix within** a generation, keeping most children structural.
  The proposer may allocate up to two objective-neutral children across all parents while the lane
  is open. Zero is a valid and usually correct choice for the proposer to make.
- `retune_stall_generations` bounds the lane **across** generations and ties it to progress:
  without it, one slot per generation forever is an unbounded parameter search running
  alongside the structural one, since each retune winner is a frontier member that can be
  retuned again next generation.

Decide whether the lane is open before proposals. It opens only if the best
measured ranking key across the frontier strictly improved at least once in the
preceding two completed generations. Record the historical values that decided it.
With no qualifying improvement, allocate zero retune slots; do not invent progress
to open the lane at startup. A single barren generation is normal early on — a generation
whose every child breaches the gate leaves the best passing ranking key untouched — and
closing the lane on one such generation would misfire exactly when margin is scarcest.

If unexpected measured neutrality exceeds the allocation, retain the real measurements
and report the unresolved policy violation. Do not retrospectively invent an allocation.

## Parent/child collapse

Beam also needs a **collapse step** that deduplication cannot provide. Deduplication keys on
task-program identity, and a retune has different source from its parent, so
both survive at an identical ranking key. Only when the retune lane is open this generation,
collapse each parent-child pair at an identical ranking key down to the passing member with the
larger gate margin. Without this step, two beam slots hold the same ranking-key
value and the frontier stops representing distinct trade-offs.

## What the logs must carry

Enough to resume the budget without chat history, and enough for a reader to audit it:

- per generation, whether the lane was open and how many slots;
- the values that decided it;
- the measured gate margin for every candidate, so the next proposer knows how much
  quality margin remains;
- `objective_neutral` per candidate, **as classified from the recorded metric**, which may
  differ from what the card declared;
- the remaining allowance after the generation.

Record these fields in `search.jsonl` and `generations.jsonl`.
Keep the frontier history in `search_state.json` sufficient to resume the same
two-generation window without chat history.

## A property, not a schedule

A retune chain accepted for larger gate margins has limited room for improvement
at the same ranking key. That is a reason to expect diminishing returns, not a
guarantee about when retuning stops.

The two-generation stall window is what closes the lane when the frontier no longer
improves its ranking key. Continue structural research while the remaining task budget
can fit a complete generation.

## Note

An accepted structural improvement may move quality toward its gate threshold;
that interval is the margin this method can spend, and the lane is how it gets
spent deliberately instead of accidentally. It cannot cross the threshold, so
every selected member stays inside the gate for the life of the run, but quality is not monotone.
