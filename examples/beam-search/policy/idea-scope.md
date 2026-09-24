# Policy: idea scope

**Our rule.** One card changes one attributable thing. Beam Search uses
`composite` scope, as configured by `idea_scope` in [settings.json](../settings.json),
and requires a single stated mechanism.

## Composite scope

A card may combine several coordinated edits when they only pay off together,
or when a mechanism needs hyperparameter adjustments to make it work. It still needs one stated
mechanism. Unrelated changes bundled for convenience are not a composite idea;
they are an unattributable one.

Composite scope trades attribution for reach. It is the right choice when the promising moves
are structural and a single-variable search cannot express them, and the wrong choice when
the point of the round is to isolate a cause. A composite card should name what a follow-up
would need in order to decompose a win.

## Attributable does not mean small

"Replace this attention variant with that one" is a single attributable idea. "Rewrite the model"
is not. The obligation is that the change be **attributable**, not that it be minor — a rule
read as "keep edits small" produces exactly the timid, constant-shaped search this policy
exists to prevent.

Where a structural change unavoidably moves a second quantity with it — a shape-dependent
learning-rate rule, a parameter count, a step time — the card names both effects and the
evaluator attributes the result to both. That is still one idea.

## Why we hold it

With one run per candidate, an accept is a single measurement. If the card changed four
unrelated things, the accept tells us the bundle was better and nothing about which part
carried it, so the next generation has no direction to push in. Attribution is not bookkeeping
here; it is the only mechanism by which generation `n+1` is better informed than generation `n`.

A genuinely interacting pair of changes, each harmful alone, can be invisible
when proposals are limited to isolated edits. Composite scope allows that pair
while requiring a single mechanism and an explanation of its interaction.

## Interaction with the retune lane

A composite card may combine a structural change with the hyperparameter
adjustments needed to make it work. If its measured ranking key changes, it is structural and
spends no retune allowance. If that key exactly equals the parent's value,
it spends a retune attempt regardless of the card's label. State this in the
proposer's brief. See [`retune-lane.md`](retune-lane.md).
