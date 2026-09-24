# Runtime

Prepares a run, starts the method's command and manages evaluation requests.
The method owns its research loop.

| Module | Responsibility |
| --- | --- |
| [lifecycle.py](lifecycle.py) | Configuration, frozen context, start/resume/stop and evaluation API |
| [environment.py](environment.py) | Private method copy, dependency setup and explicit setup retry |
| [dispatch.py](dispatch.py) | Durable requests, background workers and recovery of existing responses |
| [service.py](service.py) | Local socket delivery from methods to the active engine |
| [readiness.py](readiness.py) | Selected-run environment and optional model checks |
| [checks.py](checks.py) | Task package and method interface validation |
| [inputs.py](inputs.py) | Read captured input files |
| [source.py](source.py) | Identify the initial method source copy |

Use the [engine CLI](../README.md) and [method API](../API.md) to run experiments.

The supervisor owns a local Unix socket service while the method is running.
Experiment API commands send requests to this service, so evaluation workers
start outside the method's device sandbox. Requests arrive through the socket;
completion waits on the existing worker lease, without polling request files.
The dispatch records remain the durable request and response history.

Closing a client connection does not cancel accepted work. The service closes
when the controller exits; accepted workers keep running. Operator API commands
can execute directly while no controller owns the run, including recovery of an
existing request after stop or completion. An unavailable service during an
active controller is an error, never permission to start a worker in the client.

Setup, check, start and resume share the private method copy. The generated
environment destination must be separate from both the supplied source and run
records, including when paths pass through symlinks. A shared scratch ancestor
with separate child directories is allowed. Client identity covers the copied
`autoarena.py` module; packaging-only edits do not invalidate new setup plans.
Existing frozen records are never rewritten.

The method's `environment.check` owns validation of scripts and dependencies.
Runtime checks command rendering and executable availability without interpreting
ordinary arguments ending in `.py` or `.sh` as filenames. A missing script that
the method's check overlooks can therefore fail at controller launch.

Every new dispatch submission requires a stable `request_id`. Reusing it requires
the exact same payload. Optional `parent_id` is null or a nonblank native string;
its original text is recorded without requiring a measured parent or imposing
candidate filename syntax. Candidate IDs retain their path validation. Unknown
worker ownership remains unresolved regardless of ledger coverage.

A program remains reserved while any attempt is charged or unresolved. Resolved,
uncharged attempts permit a new candidate/request for the same code, including
after an adjudication. The original reservations and ledger rows remain intact.

Readiness, preparation and reference allocation check the task accelerator against
every device in the selected worker slice through the compute facts interface.
An empty, unknown or mismatched selection cannot start measurement. A failed GPU
readiness check also skips the measurement-runtime probe. Rejection before reference
measurement preserves successful setup and permits explicit resume after the
allocation is corrected.

`collect` returns the canonical collected-result schema unchanged. Use `status`
for controller state and in-flight requests. Completion still requires settled
evaluations; the supervisor then terminates a lingering owned controller.
