# Benchmark task and evaluation

Use the supplied task definition for its optimization target, direction, every
constraint and the allowed code changes. Scores are raw target values in the
task's units. The cached baseline is this run's canonical reference measurement.

The configured evaluation command submits the current task code to AutoArena.
Before invoking it, write `research_log.json` with a nonempty `ideas` list and
`status` describing the proposed experiment. Include node and parent identifiers
in that account. The command prints the complete canonical result, raw target
score and constraint eligibility. Preserve these observations in native reports,
including failed or ineligible candidates and what was learned from them.

Train and evaluate a candidate with:

    {eval_cmd}

Do not run `train.py` directly. Re-running the command with unchanged task
code returns the original result rather than measuring again.

This task provides one evaluation set. There is no additional B_test or final
confirmation measurement. GitMergeBranch reads the candidate and incumbent's
source-bound canonical results, checks task eligibility, then retains Arbor's
normal improvement comparison, merge guideline and Git merge. A cached result
is evidence from the original experiment, not independent confirmation.

Keep the native research tree, executor workflow, feedback, resume and final
reports. A raw-score best node can violate constraints; distinguish it from an
eligible solution when reporting. The benchmark enforces the task's measurement
budget.
