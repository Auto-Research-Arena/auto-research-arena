# Heuresis support changes

[install.py](../install.py) and [runtime.patch](runtime.patch)
apply these changes to the original Heuresis source:

| Component | Change |
| --- | --- |
| Sandbox | Forward Bedrock settings; allow writable evaluation-queue mounts and longer Claude shell timeouts. Use the host PID namespace when fresh procfs mounting fails. Map CUDA indices to device minors when needed. |
| Harness | Preserve extra mounts' read-only flags. |
| Grading/parsing | Retain details from the attempt supplying the selected score. |
| MAP-Elites feedback | Check task eligibility before archive admission and carry canonical metric metadata through native records. |
| Archive recovery | Persist/rebuild actual archive fitness rather than the displayed score. |
| Classifier logging | Record native LLM/fallback attribution for each candidate. |
| Judge | Use frozen task rules and reference code. Recover the delivered canonical stdout into native regrading instead of repeating training. |
| Iteration boundary | Prevent new proposals while benchmark delivery is unresolved; return definitive API rejection as an unsuccessful attempt. |
| Installed resources | Resolve installed templates and use short temporary paths for grading sockets. |
| Authentication | Mount Claude credentials without importing other sessions and use the configured provider model alias. |

The [task adapter](nanogpt/adapter.py) supplies task files and instructions.
The [bridge](../dispatch.py) records research and request identity and returns
canonical stdout, stderr and results.

[objective.py](nanogpt/objective.py) combines the target with a bounded tie-break
fraction. Crossing a tie-break power of ten can reverse ordering, and the fraction
can reorder nearby fractional targets. Benchmark ranking uses raw metrics instead.
[provenance.json](provenance.json) identifies the original patched files.
