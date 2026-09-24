# GPU execution

`local` uses GPUs already available on the host. The engine controls experiment
lifecycle; compute runs commands and reports facts without interpreting metrics
or charging launches.

```json
{
  "backend": "local",
  "visible_devices": [0, 1],
  "accelerator": "NVIDIA_A100_80GB",
  "mounts": {"~/.cache/autoresearch": "data/autoresearch"},
  "env": {"UV_PROJECT_ENVIRONMENT": ".local/measurement", "UV_NO_SYNC": "1"}
}
```

For local runs started through the engine, the data mount and measurement
environment shown above are defaults. Supply `mounts` or `env` to override them;
`"mounts": {}` disables the default mount. Data mounts require bubblewrap (`bwrap`).

| Key | Meaning |
| --- | --- |
| `backend` | `local` or an explicitly supplied `module:BackendClass` extension |
| `visible_devices` | Allocated device IDs as a list or comma-separated string; duplicate IDs are rejected |
| `accelerator` | Optional hardware declaration, checked against every selected device |
| `run_id` | Optional execution-allocation identity; otherwise generated per backend instance |
| `mounts` | Map task cache paths to repo data inside each process using `bwrap` |
| `env` | Measurement-runtime environment, separate from the method environment |

The worker splits the selected devices into disjoint slices. Independent runs
must be allocated disjoint GPUs by the operator. No host-wide scheduler is added.
An omitted selection uses `CUDA_VISIBLE_DEVICES`, or discovers host device indices
when that variable is absent. An explicit empty list or empty mask selects no GPUs;
it never enables host discovery. A request cannot override its worker's assigned
CUDA mask, worker ID or launch ID through `env`.
Prepare the measurement environment and dataset before launch; the benchmark
method environment is created separately. See the [root walkthrough](../../README.md).

`Backend` supplies workers and `Worker.run(request)` returns a `LaunchOutcome`.
The request carries command argv, absolute working/log paths, environment, GPU
count, launch identity and optional timeout. It carries no Git-revision gate or
candidate label. The evaluator owns source identity, immutable instruments and
accounting.

Outcomes report exit code, timestamps, elapsed time, log paths, worker/allocation
identity and whether execution started, timed out or was preempted. A command
failure returns an outcome; unavailable execution resources raise `ComputeError`.
`Worker.refused` records an unstarted command and its logs. Timeouts send SIGTERM
to the owned group, then SIGKILL after the leader exits or its grace period expires.
The leader remains unreaped until escalation so its process-group ID cannot be
reused. Interruptions terminate owned work and propagate; they do not return a
completed measurement.

Extensions use `"backend": "your_module:BackendClass"` and subclass `Backend`.
Declare extra configuration keys with `extra_config_keys`, override `make_worker`
to return your `Worker`, and override `release` for owned-resource cleanup.
`from_config(config)` is an optional construction hook. `prepare_node()` runs for
each new worker after all workers have been constructed; it is not once-per-node
or once-per-run setup. Local allocations remain operator-owned. There is no
registration decorator, preset table or separate backend-name requirement.

`backend.gpu_names(gpu_ids)` reports observed names in the exact requested ID order.
It returns `[]` when the complete selection cannot be resolved, including aliases
that select the same GPU twice. External backends may override this factual hook.
Runtime must require one observation per selected ID and compare each name with
the frozen task accelerator using `engine.compute.probe.matches`; compute does
not receive the task or enforce its scientific rules. `preflight()` takes no
arguments. `describe()` reports its configured/environment selection from one
names query; its accelerator is observed, never filled from the declaration.
Queries time out after 10 seconds; missing or incomplete observations remain unknown.

Optional conformance checks exercise CPU commands, logs, timeouts, failed process
starts and worker identities:

```bash
python3 -m engine.compute.selfcheck --config run-config.json
```

For CPU-only development, synthetic device IDs can be supplied explicitly:

```bash
python3 -m engine.compute.selfcheck --backend local \
  --backend-config '{"visible_devices":[0,1]}'
```

This runs no GPU workload and does not certify hardware readiness. Real benchmark
runs still need the selected GPUs, measurement environment and dataset.

`backend.py` defines requests, outcomes and worker allocation. `runner.py` executes
one subprocess per launch, preserves stdout/stderr, handles timeouts and drains
owned processes. `probe.py` reports local hardware. `selfcheck.py` exercises worker
execution without training. `storage.py` supplies cache paths and process-local
dataset mounts. Optional coding-agent transports belong to
[example launchers](../../examples/README.md).
