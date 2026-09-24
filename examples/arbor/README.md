# Arbor native search integration

This integration uses the [pinned Arbor source](upstream/) with native defaults
for tree depth and merge threshold. Retrieval uses the configured `alphaxiv`
backend.

[settings.json](settings.json) sets `max_cycles` to 100 instead of Arbor's native
40. This raises the method's cycle limit so it is less likely to stop before the
task's launch budget is spent. A cycle is not a launch; the engine still enforces
the task budget.

The method reads the frozen task and measured reference, owns its research tree,
and sends experiments through the canonical API. [settings.json](settings.json)
selects its model and execution settings; [interface.json](interface.json)
defines installation, readiness, startup and resume.

[install.py](install.py) adds [measurement.md](measurement.md) to both the
coordinator and executor system prompts in the private installed copy. These
instructions require task training and measurements through the evaluation
command. They are included whenever a prompt is built, including after resume,
and do not depend on the coordinator preserving them in a dataset summary.

The native coordinator and its descendants run under `bwrap` with no GPU devices
in `/dev`. The evaluation command reaches the outside engine through its local
socket, where the configured measurement worker has GPU access. Readiness checks
that bubblewrap can launch this sandbox. Native CPU work, research decisions and
network access are unchanged.

Use the pinned `upstream/` source and preserve native state when resuming.
