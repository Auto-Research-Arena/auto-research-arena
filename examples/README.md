# Example integration changes

Each unversioned `interface.json` runs setup, check, start and resume from one
persistent private copy of the method directory. Examples with a dedicated Python
runtime create a Python 3.12 `.venv` explicitly in setup and select
`.venv/bin/python` for their Python scripts. The engine exposes `autoarena` through
`PATH` and its Python client through `PYTHONPATH`; setup runs once, and resume
reuses the same files.

Claude-based launchers can opt into the shared [Claude transport](claude_transport.py)
with `python -m examples.claude_transport -- claude <native arguments>`.
This example helper keeps native background sessions alive through the final
result and idle event, preserving model, permissions, session/resume arguments,
result output and process cleanup. Beam Search, Sequential Search, GEAR and AutoScientists
use it. The existing method `PYTHONPATH` includes the repository root, so the helper
is available from the private method source directory without copying it or adding
a run binding.

Each guide describes changes relative to the original method:

- [TPE](tpe/README.md): task-bound search space, evaluation API and efficiency scores.
- [Heuresis](heuresis-map-elites/README.md): task adapter, sandbox bridge and archive feedback.
- [GEAR](gear/README.md): task binding, score projection and constraint feedback.
- [Beam Search](beam-search/README.md): API, task arithmetic and execution records.
- [Arbor](arbor/README.md): native tree and merge defaults, benchmark evaluation
  and a configured cycle limit.
- [AutoScientists](autoscientists/README.md): task profile, evaluation and coordination service.
- [Sequential Search](sequential-search/README.md): task binding, API and research records.
