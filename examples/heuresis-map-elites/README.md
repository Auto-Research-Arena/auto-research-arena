# Heuresis integration changes

Compared with [Heuresis](https://github.com/a-antoniades/Heuresis):

| Part | Original | This integration |
| --- | --- | --- |
| Task and source | Bundled NanoGPT problem, code and baseline. | [run.py](run.py) supplies the full frozen task, source and measured reference. |
| Execution | Training inside the executor workspace. | [dispatch.py](dispatch.py) bridges the sandbox to the benchmark API; training runs outside the agent workspace. |
| Logging | Native ideas and experiment notes. | Adds premeasurement research logs, durable request IDs and canonical source/result records. Records whether classification used the native LLM or keyword fallback. |
| Archive fitness | Validation-loss score. | [objective.py](support/nanogpt/objective.py) uses efficiency fitness and task eligibility. Its scalar approximation can differ from benchmark ordering. |
| Judge | Bundled baseline and verification run. | Frozen task/reference; suspicious evidence is recovered from canonical stdout for regrading, without an independent rerun. Rejected results cannot enter the archive. |
| Resume | Native experiment/archive restoration. | Reconciles pending benchmark requests and their delivery before further research. |
| Configuration | Native launch settings. | [settings.json](settings.json) selects Opus, one ideator, a limit of 100 iterations and disabled shared memory. |

[Support changes](support/README.md) cover sandbox compatibility and installed
source patches. [install.py](install.py) applies them to a private source copy.
[interface.json](interface.json) creates its Python 3.12 `.venv` explicitly and
uses `.venv/bin/python` for installation, checks and startup. The controller
locates the native `heuresis` executable and sandbox environment through this
interpreter's prefix.
