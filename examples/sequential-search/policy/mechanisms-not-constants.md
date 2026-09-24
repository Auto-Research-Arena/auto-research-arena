# Policy: enumerate mechanisms, not constants

**Our rule.** When the proposer enumerates what is available to change, it enumerates the
editable file's **mechanisms against the source**, not its declared parameters. Availability
is re-derived from the file each round, never inherited from a previous round's prose.

The benchmark says which files are mutable and says an editable
file is editable at every level. This policy guides how Sequential Search explores
the permitted changes.

## What the rule is

A named hyperparameter is a **convenience, not the boundary of the search**. A constants
block exists because a value is convenient to sweep, not because the search space is the
cross-product of those constants.

When the task names `train.py` as mutable, the whole file is in scope:

- the model definition, and its architecture;
- the attention implementation;
- the optimizer, and the learning-rate schedule as a mechanism rather than as a number;
- the loss;
- the data path and what the model is shown;
- the training loop, and the structure of any of the above.

Replacing a mechanism, adding one, and deleting one are all legitimate candidates. So is
changing a constant — the rule is about where the *enumeration* looks, not about what is
permitted.

## Why we hold it

Avoid spending tens of consecutive rounds on hyperparameter tuning only,
where the search enumerates only the constants at the top of the file and
never notices a productive axis in the model code below it.

Nothing may appear wrong: every round has a card, a mechanism, a measurement
and a decision. Yet the search can remain confined to a narrow set of constants,
leaving productive changes elsewhere in the model code unexplored.

This is an agent-behaviour failure, not a search-algorithm failure. Neither a wider beam nor a
better acquisition function fixes it, because every candidate any of them proposes comes out
of the same enumeration. It is fixed in the brief or not at all — which is precisely why it
belongs in a policy file and not in an objective.

## What it means operationally

Two obligations on the proposer's brief:

1. **Enumerate against source.** The brief carries, or the proposer reads, the current
   editable file. Not a summary of it, and not last round's list of options.
2. **Re-derive availability each round.** A mechanism ruled out three rounds ago was ruled
   out against a different incumbent. Inheriting that judgement is how a search narrows
   monotonically until only constants remain.

The corresponding failure to watch for in the log: a run whose cards over several consecutive
rounds all name quantities from the same block of the file. That pattern is the symptom, and
it is visible in `search.jsonl` before it is visible in the results.
