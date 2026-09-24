"""Bind Heuresis MAP-Elites to the engine API.

The adapter supplies task code, rules and reference feedback, exposes the
dispatch queue and returns canonical evidence to the native research loop.
The engine performs GPU measurements; native research sandboxes have no device.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from heuresis import HackerJudge, Harness, Mount, Workspace
from heuresis.tasks.adapter import TaskAdapter

if TYPE_CHECKING:
    from heuresis.qd import FeatureClassifier
from heuresis.tasks import baseline_scores
from heuresis.tasks.nanogpt import NanoGPTGrader
from heuresis.tools.defaults import MEMORY

_TASK_DIR = Path(__file__).resolve().parent

_SEED_ENV = "AUTOARENA_SEED_DIR"
_QUEUE_ENV = "AUTOARENA_QUEUE_DIR"
_DISPATCH_ENV = "AUTOARENA_DISPATCH_SH"


def _env_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(f"{name} is required to locate this run's task or workspace")
    return Path(raw)


def seed_dir() -> Path:
    return _env_path(_SEED_ENV)


def _seed_files() -> dict[str, Path]:
    root = seed_dir()
    return {"train.py": root / "train.py", "prepare.py": root / "prepare.py"}


class NanoGPTAdapter(TaskAdapter):
    name = "nanogpt"
    always_inherits_parent = False

    # All GPU work happens through Benchmark.evaluate and the engine. Nothing in
    # this process or sandbox opens a device, so the loop must not reserve one:
    # `gpu_slice` becomes empty, `reserve_gpus` no-ops, and no nvidia device is bound.
    uses_gpu = False

    def __init__(self) -> None:
        self._problem = ""
        self._idea_schema: str | None = None
        self._metric: str | None = None
        self._baseline: float | None = None
        # Filled from baseline_scores.yaml in setup_objective; defaults chosen so a
        # missing file cannot silently invert the direction of improvement.
        self.metric_label = "ranking_key"
        self.lower_is_better = True

    @property
    def task_dir(self) -> Path:
        return _TASK_DIR

    @property
    def runs_root(self) -> Path:
        # Native records live under this run's scratch directory.
        return _env_path("AUTOARENA_RUNS_ROOT")

    @property
    def problem_text(self) -> str:
        return self._problem

    @property
    def idea_schema_text(self) -> str | None:
        return self._idea_schema

    @property
    def metric(self) -> str | None:
        return self._metric

    @property
    def baseline(self) -> float | None:
        return self._baseline

    @property
    def task_prompt_template(self) -> str | None:
        return "heuresis/tasks/nanogpt/prompts/ideator_task.j2"

    # ---- cell-search capability (SupportsCellSearch) ---------------------
    @property
    def cell_ideator_prompt(self) -> Path:
        return self.strategy_prompt("cell")

    @property
    def objective_label(self) -> str:
        from heuresis.tasks.nanogpt import objective as _obj

        return _obj.objective_label()

    def make_classifier(self) -> "FeatureClassifier":
        from heuresis.tasks.nanogpt.features import make_classifier

        return make_classifier(use_llm=True)

    def make_objective(self, settings: Any):
        """The task's gated objective. Never None -- the quality gate is explicit."""
        del settings
        from heuresis.tasks.nanogpt import objective as _obj

        _obj.load_ranking()  # raises loudly if the task is not this shape
        print(f"Objective: {_obj.objective_label()}")
        print(f"Task: {_obj.task().task_id}  run: {_obj.run_dir()}")

        def _evaluate(info: dict, workspace: Any):
            return _obj.evaluate(info, workspace)

        return _evaluate

    def stop_requested(self) -> bool:
        # Finish current native feedback, then pause before another proposal after
        # uncertain delivery or invalid request metadata. Never select a substitute.
        return (Path(os.environ["AUTOARENA_NATIVE_STATE"]).parent / "pause-required.json").exists()


    def normalize_settings(self, settings: Any) -> None:
        # Use one ideator; the engine assigns GPUs for benchmark evaluations.
        settings.gpus = []
        settings.num_ideators = 1

    def preflight(self, settings: Any) -> list[str]:
        from heuresis import preflight
        errors = preflight.check_agent(settings.agent)
        if error := preflight.check_bwrap():
            errors.append(error)
        return errors

    def setup_objective(self, settings: Any) -> None:
        del settings
        scores = baseline_scores(_TASK_DIR)
        self._metric = scores.get("metric")
        self._baseline = scores.get("baseline")
        self.metric_label = str(scores.get("metric") or "ranking_key")
        self.lower_is_better = scores.get("objective", "min") == "min"

    def on_experiment(self, exp: Any, state: Any, settings: Any) -> None:
        Path(os.environ["AUTOARENA_NATIVE_STATE"]).write_text(
            __import__("json").dumps({"experiment_id": exp.id}) + "\n")
        del state, settings
        from heuresis.tasks.nanogpt import objective as _obj

        self._problem = (_TASK_DIR / "problem.j2").read_text()
        # problem.j2 is injected as the `{{ problem }}` VALUE of another template, so its
        # own text is never Jinja-rendered: a `{{ ... }}` inside it would reach the agent
        # literally. Placeholders are substituted here, from the task, so the prompt
        # cannot state a stale gate, a stale ceiling or a stale baseline.
        gate_value, gate_operator = _obj.gate_bound()
        substitutions = {
            "__TASK_ID__": _obj.task().task_id,
            "__GATE_METRIC__": _obj.gate_metric(),
            "__GATE_OP__": gate_operator,
            "__GATE__": repr(gate_value),
            "__RANK_METRIC__": _obj.ranking_metric(),
            "__TIEBREAK_METRIC__": str(_obj.tiebreak_metric() or "(none)"),
            "__OBJECTIVE__": _obj.objective_label(),
            "__BASELINE__": repr(self._baseline),
        }
        for key, value in substitutions.items():
            self._problem = self._problem.replace(key, value)
        leftover = [k for k in substitutions if k in self._problem]
        if leftover:  # pragma: no cover - defensive
            raise RuntimeError(f"problem.j2 still carries {leftover}")
        schema = _TASK_DIR / "idea_schema.md"
        if schema.exists():
            self._idea_schema = schema.read_text()

    def executor_task_vars(self, **kwargs: Any) -> dict[str, Any]:
        """Describe benchmark-managed GPU execution in the executor prompt."""
        base = super().executor_task_vars(**kwargs)
        base["gpu_info"] = "Benchmark-managed GPU; one launch per dispatch"
        return base

    def make_judge(self, settings: Any) -> HackerJudge | None:
        if not settings.enable_judge:
            return None
        jh = Harness(settings.judge_agent, model=settings.judge_model, gpus=[])
        return HackerJudge(jh, _TASK_DIR, baseline_dir=seed_dir(), timeout=settings.judge_timeout)

    def _memory_tools(self, memory_on: bool) -> list:
        return [MEMORY] if memory_on else []

    def seed_files(self) -> dict[str, Path]:
        return _seed_files()

    def parent_files(self, parent_run: Any) -> dict[str, Path]:
        # Inherit the parent's training code and the task's immutable instrument.
        src = Path(parent_run.workspace)
        files = {"train.py": src / "train.py",
                 "prepare.py": seed_dir() / "prepare.py"}
        # Pass the selected parent's candidate ID to the dispatcher.
        marker = src / "arena_result.json"
        if marker.is_file():
            files["arena_parent.json"] = marker
        return files

    def ideator_workspace(self, settings: Any, *, prompt: Path) -> Workspace:
        return Workspace(
            tools=self._memory_tools(settings.memory),
            files=_seed_files(),
            venv=Path(os.environ["AUTOARENA_NATIVE_VENV"]),
            prompt=prompt,
            role="ideator" if settings.memory else None,
        )

    def executor_workspace(
        self, *, files: dict[str, Path], memory_on: bool, prompt: Path | None = None
    ) -> Workspace:
        return Workspace(
            tools=self._memory_tools(memory_on),
            files=files,
            prompt=prompt or (_TASK_DIR / "executor_prompt.j2"),
            # Reuse the method environment for editing, dispatch and grading.
            venv=Path(os.environ["AUTOARENA_NATIVE_VENV"]),
            requirements=None,
            role="executor" if memory_on else None,
            # Make train.py editable in the research workspace.
            editable="train.py",
            lock_down_edits=True,
        )

    def make_grader(self, exec_dir: Path) -> NanoGPTGrader:
        return NanoGPTGrader(exec_dir / ".grade.sock")

    def mounts(self) -> list[Mount]:
        """Mount the read-only dispatcher and this run's writable queue."""
        queue = _env_path(_QUEUE_ENV)
        dispatch = _env_path(_DISPATCH_ENV)
        return [
            Mount(source=dispatch, target="/arena/dispatch.py", readonly=True),
            Mount(source=queue, target="/arena/queue", readonly=False),
        ]
