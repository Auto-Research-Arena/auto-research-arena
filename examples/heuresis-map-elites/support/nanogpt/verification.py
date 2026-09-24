"""Retrieve original benchmark evidence for the native judge's regrading step."""

from __future__ import annotations

import json
import os
from pathlib import Path


def restore_evidence(exec_workspace: Path) -> bool:
    """Restore canonical stdout without evaluating or changing executor evidence.

    Native judging may distrust the executor's copy of run.log. Read the original
    benchmark stdout through its host-owned response, preserving the submitted
    log for review. This checks evidence authenticity, not repeatability.
    """
    workspace = Path(exec_workspace)
    queue = Path(os.environ["AUTOARENA_QUEUE_DIR"])
    token_path = workspace / ".arena_request_id"
    if not token_path.is_file():
        return False
    token = token_path.read_text().strip()
    result_path = queue / "done" / token / "arena_result.json"
    if not result_path.is_file():
        # A request that was rejected before it reached a GPU has no canonical stdout to
        # restore. Nothing to regrade, so the judge keeps the evidence it has; this is a
        # missing file, not a finding about the workspace.
        return False
    result = json.loads(result_path.read_text())
    regenerated = workspace / "regenerated"
    regenerated.mkdir(exist_ok=True)
    (regenerated / "run.log").write_bytes(Path(result["logs"]["stdout"]).read_bytes())
    return True
