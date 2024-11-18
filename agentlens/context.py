from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, TypeVar

import petname

from agentlens.evaluation import Hook, Mock
from agentlens.provider import InferenceCost
from agentlens.stack import ContextStack
from agentlens.utils import now

T = TypeVar("T")

_run_context: ContextVar[Run | None] = ContextVar("run_context", default=None)
_iteration_context: ContextVar[int | None] = ContextVar("iteration", default=None)


@dataclass
class Observation:
    id: str
    dir: Path
    start_time: datetime = field(default_factory=now)
    end_time: datetime | None = None
    iteration: int | None = None
    cost: InferenceCost = field(default_factory=InferenceCost)

    def end(self) -> None:
        self.end_time = now()


class Task(Observation): ...


class Run(Observation):
    def __init__(self, runs_dir: Path, task_name: str):
        id = self._create_run_id()
        super().__init__(id=id, dir=runs_dir / id)
        self.contexts = ContextStack[dict[str, Any]]("contexts")
        self.hooks = ContextStack[dict[str, list[Hook]]]("hooks")
        self.mocks = ContextStack[dict[str, Mock]]("mocks")
        self.observations = ContextStack[Observation]("observations")

        # Create and push root task
        root_task = Task(
            id=task_name,
            dir=self.dir / task_name,
        )
        self.observations._stack.set([root_task])

    def _create_run_id(self) -> str:
        timestamp = now().strftime("%Y%m%d_%H%M%S")
        key = petname.generate(words=3, separator="_")
        return f"{timestamp}_{key}"

    @staticmethod
    def current() -> Run | None:
        return _run_context.get()

    @staticmethod
    def current_iteration() -> int | None:
        return _iteration_context.get()

    @staticmethod
    def set_iteration(i: int | None) -> None:
        _iteration_context.set(i)

    @staticmethod
    def current_task() -> Task:
        run = Run.current()
        if run is None:
            raise ValueError("No active run context")
        task = run.observations.current
        if not isinstance(task, Task):
            raise ValueError("No active task context")
        return task

    @staticmethod
    def start(runs_dir: Path, task_name: str) -> Run:
        run = Run(runs_dir, task_name)
        run.dir.mkdir(parents=True, exist_ok=True)
        _run_context.set(run)
        return run

    @staticmethod
    def end():
        _run_context.set(None)

    @contextmanager
    def create_observation(self, name: str) -> Iterator[Observation]:
        parent = self.observations.current
        if parent is None:
            raise ValueError("No active task context")

        idx = self.current_iteration()
        observation = Task(  # Note: creating Task instead of Observation
            id=name,
            iteration=idx,
            dir=parent.dir / (name + (f"_{idx}" if idx is not None else "")),
        )
        observation.dir.mkdir(parents=True, exist_ok=True)

        with self.observations.push(observation):
            try:
                yield observation
            finally:
                observation.end()
