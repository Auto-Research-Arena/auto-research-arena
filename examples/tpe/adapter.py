"""Connect the upstream TPE Runner to AutoArena at the experiment boundary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from tpe_scoring import scalar

ROOT = Path(__file__).resolve().parent


def native():
    from autoresearch_automl.core.runner import Runner, RunConfig
    from autoresearch_automl.backends.optuna_backend import OptunaBackend
    from autoresearch_automl.core.search_space import SearchSpaceBuilder, KNOWN_HP_METADATA
    if "baseline_config" not in RunConfig.__dataclass_fields__:
        raise RuntimeError("Install the pinned TPE package with runtime.patch first")
    return Runner, RunConfig, OptunaBackend, SearchSpaceBuilder, KNOWN_HP_METADATA


def record_once(path, value):
    """Keep a stable API payload for this candidate request."""
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Native trial inputs changed: {path}")
        return
    with path.open("x") as stream:
        stream.write(text)
        stream.flush()


class ArenaExperiments:
    """The native Runner still samples, snaps, records, feeds back and stops."""

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.directory = benchmark.workspace / "tpe"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.original_source = (benchmark.task_dir / "code/train.py").read_text()

    def stage(self, trial_id, code):
        """Reuse identical trial code; give a resumed redraw its own candidate ID."""
        stem = f"{trial_id:04d}"
        attempt = 0
        while True:
            suffix = stem if attempt == 0 else f"{stem}-r{attempt}"
            source = self.benchmark.workspace / "candidates" / f"tpe-trial-{suffix}"
            if not source.exists():
                shutil.copytree(self.benchmark.task_dir / "code", source)
                (source / "train.py").write_text(code)
                return suffix, source
            if (source / "train.py").read_text() == code:
                return suffix, source
            attempt += 1

    def run(self, config):
        from autoresearch_automl.core.config_injector import ConfigInjector
        from autoresearch_automl.core.experiment import ExperimentOutcome
        from autoresearch_automl.core.objective import ExperimentResult

        benchmark = self.benchmark
        if config.trial_id == 0:
            result = benchmark.reference_result
        else:
            code = ConfigInjector(benchmark.task_dir / "code/train.py").inject(config.hp_config)
            suffix, source = self.stage(config.trial_id, code)
            identifier = f"tpe-trial-{suffix}"
            research = {"ideas": [{"parameters": config.hp_config}],
                        "status": "Native TPE trial ready", "trial_id": config.trial_id}
            record_once(self.directory / f"request-{suffix}.json",
                        {"request_id": identifier, "source": str(source), "research_log": research})
            result = benchmark.evaluate(source, identifier, candidate_id=identifier, research_log=research)
        # The shared reference-normalized score occupies the native single objective.
        # All unmodified canonical observations remain attached to the native record.
        value, label = scalar(result, benchmark.task, benchmark.reference_result)
        metrics = result.get("metrics", {})
        observation = ExperimentResult(
            val_bpb=value,
            peak_memory_gb=(metrics["peak_vram_bytes"] / 2**30
                            if metrics.get("peak_vram_bytes") is not None else None),
            wall_time_seconds=metrics.get("training_seconds"),
            success=result["status"] == "ok",
            error=None if result["status"] == "ok" else result["status"],
            extra_metrics={"canonical_result": result, "scalar_classification": label},
        )
        return ExperimentOutcome(config=config, result=observation)


def search(benchmark, *, max_trials, resume=False):
    import optuna

    Runner, RunConfig, Backend, Builder, metadata = native()
    seed = 0
    execution = ArenaExperiments(benchmark)
    records = execution.directory / "trials.jsonl"
    if records.exists() and not resume:
        raise RuntimeError("Native state exists; use --resume")
    baseline = {hp.name: hp.value for hp in Builder(benchmark.task_dir / "code/train.py").extract_hps()
                if hp.name in metadata}
    budget = benchmark.task["launch"]["train_seconds"]
    sampler = optuna.samplers.TPESampler(seed=seed)
    config = RunConfig(
        train_py_path=benchmark.task_dir / "code/train.py", backend=Backend(sampler=sampler),
        n_trials=max_trials + 1, budget_min=budget, budget_max=budget,
        seed=seed, objectives=["val_bpb"], results_dir=execution.directory,
        time_budget=0, resume=resume, search_space_include=list(metadata), baseline_config=baseline,
    )
    runner = Runner(config, experiment_runner=execution)
    summary = runner.run()
    benchmark.finish(f"Native TPE Runner completed {max_trials} candidate trials")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.check:
        native()
        from autoarena import Benchmark
        print("Upstream TPE Runner and AutoArena client ready")
        return
    from autoarena import Benchmark
    settings = json.loads((ROOT / "settings.json").read_text())
    search(Benchmark.from_context(), max_trials=settings["max_trials"], resume=args.resume)


if __name__ == "__main__":
    main()
