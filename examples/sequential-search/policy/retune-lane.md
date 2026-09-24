# Policy: the retune lane

**Our rule.** An objective-neutral candidate — one whose ranking key lands at exactly its
parent's value — is a legitimate move when it passes the task constraints, because
it can buy gate margin. It gets a
**bounded budget, per incumbent, counting attempts rather than accepts**, and the budget only
refills on structural progress.

**Method retention rule.** For an eligible candidate, accept a strict improvement
of the target. On an exact target tie, while retune allowance remains, accept if
either the task's official tiebreak improves or the measured quality-gate margin
strictly improves. Otherwise reject. Every exact-target tie consumes a retune
attempt, regardless of acceptance. This does not change the task's constraints,
objective or launch budget.

## What "objective-neutral" means, and how it is decided

A candidate is objective-neutral when its **ranking key** lands at exactly the
value its parent recorded, byte-exact at full recorded precision.

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
improve the incumbent’s quality-gate margin, leaving more room for a later structural change
that might worsen quality.

It cannot rescue anything. A candidate that fails the quality gate or another
task constraint is inadmissible and cannot replace the incumbent. The lane
fires only on candidates that were admissible in the first place. What survives a rejection is
the *mechanism*, which lives in the card and the evaluator's notes; a later round may re-apply
it to an incumbent that has since banked more margin. Buying that margin ahead of the attempt
is the whole point.

## Why it is bounded, and why a human sets the bound

The proposer faces an incentive it cannot see past. An objective-neutral candidate **cannot
worsen the ranking key** — it is a risk-free bid — so a round-scoped agent maximising its own
round prefers it to a structural attempt that might breach the gate. Round after round of that
is exactly the failure where a run spends its budget tuning numbers, and it arrives through a
sequence of individually defensible decisions.

The ceiling cannot be delegated to the proposer, for a structural reason rather than a
motivational one: roles are fresh subagents per round, so no proposer remembers what the
previous one chose. **A budget nobody can remember is not a budget.** It is a number in the
config, and the remaining balance is passed into every brief.

## Single incumbent

`retune_slots_per_incumbent` is four in [settings.json](../settings.json).

- **Per incumbent, not per run**, and it counts **attempts**: every objective-neutral round
  spends one whether it wins or loses.
- A **structural accept refills it** — that is, a strict improvement of the ranking key,
  the structural progress that resets this allowance. A structural reject changes nothing.
- With no budget left, the proposer is told so and proposes a structural idea. A round that
  would exceed the budget is refused rather than compared.

Sequential Search runs one idea per round, so an objective-neutral round is a round
with **no structural attempt at all**. That is what stops the lane becoming a parameter search: a run that stops
making structural progress runs the lane dry and cannot reopen it by tuning.

## What `search.jsonl` must carry

Enough to resume the budget without chat history, and enough for a reader to audit it:

- per round, whether the lane was open and how many slots;
- the values that decided it;
- the measured gate margin every round, so the next proposer knows how much
  quality margin remains;
- `objective_neutral` per candidate, **as classified from the recorded metric**, which may
  differ from what the card declared;
- the remaining balance after the round.

## A property, not a schedule

A retune chain accepted for larger gate margins has limited room for improvement
at the same ranking key. That is a reason to expect diminishing returns, not a
guarantee about when retuning stops. Ties accepted through the official tiebreak
need not improve gate margin; the per-incumbent retune limit applies to both.

When it is exhausted, continue structural research until the task's launch
budget is spent.

## Note

An accepted structural improvement may move quality toward its gate threshold;
that interval is the margin this method can spend, and the lane is how it gets
spent deliberately instead of accidentally. It cannot cross the threshold, so the
incumbent stays inside the gate for the life of the run, but it is not monotone.
