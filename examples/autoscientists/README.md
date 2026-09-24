# AutoScientists

[interface.json](interface.json) creates a Python 3.12 `.venv` in the private
method directory, installs `requests`, `pyyaml` and ClawInstitute 0.1.3 there,
and runs `launch.py` with `.venv/bin/python`. The launcher reads the adjacent
`upstream/` copy and finds ClawInstitute under that Python environment's prefix.

The private service and native workspace live under the engine-supplied `TMPDIR`,
inside the run's persistent method environment. Use `scratch_root` in the run
configuration to select storage for that environment.
ClawInstitute's stdout and stderr are appended to `service.log` in its private
service directory. Startup errors include that path so the service output can
be inspected.

Changes from the official AutoScientists repository:

- Its training task and initial baseline measurement are replaced by the frozen
  task definition, source and measured reference. The initial champion uses the
  native `metric_name`/`metric_value` format; run-local readers use that same field.
- Each native experiment agent replaces direct training with `evaluate.py`,
  which records its request identity, submits its research log and returns raw
  canonical results plus task eligibility. Interrupted requests recover by ID.
- KEEP retains strict improvement against the current champion, with task
  eligibility added. The winning agent publishes the champion as specified by
  ROLE-GPU; the orchestrator observes publication and generates follow-ups.
- Only the `0.001` auto-bracketing trigger uses reference-normalized improvement.
  Raw scores, noise checks and follow-up proposal construction are unchanged.
- Native multi-seed confirmation remains. If a task prohibits a requested seed,
  confirmation is recorded as blocked and the candidate is not promoted.
- The outer Claude model is configured explicitly. The example uses two
  measurement workers and stops on zero KEEPs in the last ten experiments or
  the task's launch cap, whichever comes first.
- A private ClawInstitute service replaces the pre-existing service. Native
  same-run state is retained; scientific files, shared workspace revisions and
  discussion posts are exported before shutdown, excluding credentials and raw
  agent sessions. The service exposes at most 100 comments per workspace.
