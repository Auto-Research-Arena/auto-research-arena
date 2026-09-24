# Beam Search

Method ID: `beam-search`.

Beam Search maintains a frontier of up to four candidates. It starts with the
measured reference, freezes the frontier for each generation and proposes two
children per parent. A persistent driver creates fresh proposer, executor and
evaluator sessions with the task, parent sources, research history and
[active policies](policy/README.md).

[prompt.md](prompt.md#one-generation) defines the complete generation:

| Part | Rule |
| --- | --- |
| Children and workers | Two children per frontier member, interleaving parents in the submission order. Four measurement workers; the API leases whichever worker is available. |
| Retune allowance | At most two objective-neutral children across the generation, available only when the best frontier ranking key improved within the preceding two completed generations. |
| Selection | Pool the frozen parents and successfully measured children, deduplicate their task-program identities and exclude ineligible entries. Only when the retune lane is open this generation, collapse exact-target parent/child pairs to the passing member with larger gate margin. Then keep the best four by the task's canonical rank. |
| Remaining ties | Only on an exact remaining rank tie, prefer lower complexity, then lower resource usage, then the older member, using available evidence. |
| Generation boundary | Complete all children and feedback before publishing the next frontier. If the remaining budget cannot fit two children per frontier member, record that reason and stop without selecting a partial generation. |

[selection.py](selection.py) computes eligibility, program identity and ranking
values from the task and canonical results. The evaluator owns retune collapse
and frontier selection.

The driver maintains `search_state.json`, `search.jsonl`, `PROGRESS.md` and
`generations.jsonl`, preserving cards, role outputs, request identities and
complete generation transitions. Resume reconciles the unfinished generation
before making new proposals. See [Records and continuation](prompt.md#records-and-continuation)
for each file's contents, journal timing and recovery.

[settings.json](settings.json) records the model and search settings.
[interface.json](interface.json) defines installation, readiness, startup, resume
and four measurement workers. Readiness verifies that the benchmark client and
selection helper can be imported and that `claude --version` succeeds.
