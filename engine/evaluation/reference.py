"""Measure the required reference once and check its deterministic assertions."""

from pathlib import Path
from typing import List


def _reference_launch(run, task, worker, source_root: Path, backend_name: str):
    from engine.evaluation.execute import execute

    ledger = run.ledger()
    # Allocate fresh log paths without overwriting earlier ledger evidence.
    # The runtime decides whether reference initialization is allowed.
    seq = ledger.next_launch_seq()
    return execute(
        task=task,
        ledger=ledger,
        worker=worker,
        candidate_id="reference",
        source_root=source_root,
        log_dir=run.log_dir(seq),
        launch_seq=seq,
        is_reference_launch=True,
        backend_name=backend_name,
    )


def _reference_mismatches(task, metrics) -> List[str]:
    out = []
    for metric, want in (task.reference_launch.get("assert_exact") or {}).items():
        got = metrics.get(metric)
        if got is None or float(got) != float(want):
            out.append(f"{metric}: expected {want}, observed {got}")
    return out
