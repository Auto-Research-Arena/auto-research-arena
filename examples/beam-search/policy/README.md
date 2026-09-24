# Beam Search policies

These policies define Beam Search's research behavior and must be followed
throughout the run. The task definition supplies the benchmark's constraints,
metrics, ranking and budget; these policies cannot override them.

The proposer, executor and evaluator receive these policies each generation:

| Policy | Rule |
| --- | --- |
| [Idea scope](idea-scope.md) | Coordinate edits around one attributable mechanism. |
| [Mechanism enumeration](mechanisms-not-constants.md) | Derive available changes from each parent's source; the constants block is not the search space. |
| [Predictions](predict-discipline.md) | Predict source-derivable quantities and explain measured outcomes without numerical guesses. |
| [Retune attempts](retune-lane.md) | Bound objective-neutral children within each generation and keep the lane open only while the frontier improves. |

[settings.json](../settings.json) records the active settings.
