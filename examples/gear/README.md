# GEAR

Method ID: `gear`.

GEAR searches a population of programs through mutation and crossover. This
integration uses GEAR-Fixed: its [controller](upstream/gear.py) selects parents and
operators and decides which candidates enter the population. The coding agent
implements each suggestion, evaluates the candidate and records feedback.

Compared with [genetic-autoresearch](https://github.com/nm-le/genetic-autoresearch):

| Part | Original | This integration |
| --- | --- | --- |
| Task | Training project and baseline. | Frozen task code, complete task definition and measured reference. |
| Execution | Agent runs training directly. | API submissions with candidate snapshots, research logs and recoverable request IDs. |
| Feedback | Controller optimizes validation BPB. | [score.py](score.py) puts the normalized efficiency target in the native `val_bpb` field. Actual quality remains in canonical metrics. |
| Score units | BPB, memory and parameter measurements. | Every target is divided by this run's positive, finite measured reference value. Memory becomes GiB and parameters become millions; an unobserved controller slot receives `0.0`. |
| Constraints | Promotion uses native score thresholds. | [constraints.patch](constraints.patch) excludes candidates that fail task constraints from promotion and positive parent rewards, while retaining measurements and attempt counts. Eligible candidates retain native rules. |
| Stopping | Runs until interrupted. | The agent uses the task's full launch budget, completes feedback, then calls `autoarena finish`. |

[prompt.md](prompt.md) instructs the coding agent to create its workspace, apply
the supplied controller patch, and execute the research loop through the benchmark API.
Candidate task files
are kept separate from native Git/controller records. [interface.json](interface.json)
uses Claude transport and resumes the same native session. The constraint patch
applies to the workspace controller copy; the supplied original remains unchanged.
The interface selects Opus 5 using `us.anthropic.claude-opus-5[1m]`; set both
launch commands to the model ID your Claude service accepts.
