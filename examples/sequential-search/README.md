# Sequential Search

Method ID: `sequential-search`.

Sequential Search maintains one incumbent and evaluates one child per round.
A persistent driver creates fresh proposer, executor and evaluator sessions
with the task, parent source, research history from `search.jsonl` and
`PROGRESS.md`, and [active policies](policy/README.md).

[prompt.md](prompt.md) defines the research loop. Proposals coordinate edits
around one mechanism. Each structural incumbent has four objective-neutral
retune attempts. Within that allowance, an eligible exact-target tie replaces its
parent if either the task's official tiebreak improves or the measured quality-gate
margin strictly improves.

The method submits candidates through the benchmark API, records cards before
measurement and saves evaluator feedback, request identities and incumbent
state. Resume restores the incumbent and retune count from `search_state.json`,
recovers cards and history from `search.jsonl` and `PROGRESS.md`, and completes
outstanding API requests and feedback before making new proposals.

The driver maintains `search_state.json` for the incumbent and retune usage,
`search.jsonl` for round transitions and `PROGRESS.md` for the journal.

[settings.json](settings.json) records the model and search settings.
[interface.json](interface.json) defines installation, readiness, startup,
resume and one measurement worker. Complete rounds continue until the task's
launch budget is spent.
