# Policy: predict the derivable quantity, never the measured one

**Our rule.** Where the task's ranking key is derivable from source, a card must
state its exact predicted value before any GPU time is spent. Otherwise, do not
predict a value for the ranking key. Do not state an expected value, delta,
range or point estimate for the gated quality metric. Argue the mechanism for
quality; predict the arithmetic.

## The asymmetry, and why it is the whole rule

The two quantities are not the same kind of thing.

- **For shape-derived targets, the ranking key can be computed from the diff.** Parameter count
  and KV-cache size can be derived from the task's definitions
  and the candidate's shapes. A proposer that cannot compute its own candidate's
  value has not finished reading its own edit. So the
  prediction is cheap, and it is checkable **before** the launch: the executor compares the
  card's number against the instrument's and a mismatch is a defect in the card.
- **`val_bpb` is not derivable from anything.** It is an outcome of training;
  completed training steps can vary even across identical programs. A predicted value for it is
   a guess dressed as an argument, and once written it is something the candidate can be judged
  against, which is how a search starts selecting for cards that predict safely.

## What each half buys

**The predicted ranking key is a checksum, not evidence of understanding.** It confirms the
edit changed what the card said it would change, and nothing more. So a matching
prediction licenses the launch; it never licenses the argument.

**The ban on predicting quality forces a mechanism.** With no number to offer, the only thing a
card can say is what the edit does to the model and why that is worth a benchmark measurement. This is the
half that changes what gets proposed, because "share one MLP across each adjacent layer pair,
because the residual stream still receives eight transformations' worth of depth and only the
weights are reused" survives the rule, while "narrow the hidden width, expect about +0.004"
does not — the first names a mechanism, the second names a dose.

## Operationally

1. Where the ranking key is derivable, every card carries `predicted_<ranking_key>`
   as an exact integer, computed by hand from the diff, with the arithmetic shown.
   Otherwise use named predictions for its derivable factors as specified below.
2. The executor checks it against the instrument's own value before comparing anything, and
   records the mismatch as a card defect rather than as a result.
3. No card, motivation, hypothesis, `expected_effect` or risk note states an expected
   `val_bpb`, a delta, a range, or a bound.
4. Recording the **measured** gate margin for every candidate is required and is the opposite of a
   prediction: see [`retune-lane.md`](retune-lane.md), which spends that margin.

## Why we hold it

The rule shapes what a proposer is willing to write down. Requiring exact
arithmetic makes it account for what its edit changes; forbidding quality
guesses makes it explain why the change could work. The intent is to keep the
search open to structural ideas whose quality effects must be measured.

See also [`idea-scope.md`](idea-scope.md): the scope decides how much one card may
reach for, and this file decides what it must and must not claim about the reach.

## When the ranking key is not derivable

For example, updates to target, tokens to target and training time to target
depend on when training crosses a quality threshold. That crossing cannot be
computed from the diff. Requiring an exact prediction for these keys would demand the same
kind of quality guess that this policy forbids, expressed under the ranking
key's name.

Where the ranking key depends on a measured training outcome, predict its
**derivable factors** instead, and predict nothing about the key.

- For a token-count target that depends on the number of training updates,
  `tokens_to_target = updates_to_target × tokens_per_step`. The diff determines
  `tokens_per_step`, but cannot determine how many updates will reach the quality
  threshold. One factor is derivable and one is not.
  `predicted_tokens_per_step` is mandatory, exact and accompanied by arithmetic.
  Predict nothing about the number of updates or resulting trained-token total.
- Where no factor of the ranking key is derivable, predict every shape-derived
  quantity the diff touches: at minimum tokens per optimizer update, parameter
  counts (total and active, where applicable), applicable FLOP counts, gradient
  accumulation steps where present, and
  any shape the edit changes. Show the arithmetic for each named prediction.
- No card, motivation, hypothesis, `expected_effect` or risk note states an
  expected quality, time to target, updates to target, tokens to target,
  completed training steps or total trained tokens, or a delta, range or bound
  on any of them.

The executor still compares each predicted integer against the instrument's own
value before comparing anything, and a mismatch is still a card defect rather
than a result. Predicting a derivable factor checks whether the edit changed
what the card said it would change; it does not license a numerical guess about
an outcome that must be measured.
