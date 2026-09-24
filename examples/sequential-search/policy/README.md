# Sequential Search policies

These policies define Sequential Search's research behavior and must be followed
throughout the run. The task definition supplies the benchmark's constraints,
metrics, ranking and budget; these policies cannot override them.

The proposer, executor and evaluator receive these policies each round:

| Policy | Rule |
| --- | --- |
| [Idea scope](idea-scope.md) | Coordinate edits around one attributable mechanism. |
| [Mechanism enumeration](mechanisms-not-constants.md) | Derive available changes from the current source; the constants block is not the search space. |
| [Predictions](predict-discipline.md) | Predict source-derivable quantities and explain measured outcomes without numerical guesses. |
| [Retune attempts](retune-lane.md) | Objective-neutral candidates are useful but get a bounded, per-incumbent budget. |

[settings.json](../settings.json) records the active settings.
