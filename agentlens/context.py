from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar, cast

import petname

from agentlens.hooks import Hook
from agentlens.provider import InferenceCost
from agentlens.utils import now

T = TypeVar("T")

_run_context: ContextVar[Run | None] = ContextVar("run_context", default=None)


@dataclass
class Observation:
    id: str
    dir: Path
    start_time: datetime = field(default_factory=now)
    end_time: datetime | None = None

    def end(self) -> None:
        self.end_time = now()


class ContextStore:
    run: ContextVar[Run | None] = ContextVar("run", default=None)
    contexts: ContextVar[dict[str, Any]] = ContextVar("contexts", default={})
    hooks: ContextVar[dict[str, list[Hook]]] = ContextVar("hooks", default={})
    mocks: ContextVar[dict[Callable, Callable]] = ContextVar("mocks", default={})


class Task(Observation): ...


class Run(Observation):
    def __init__(self, runs_dir: Path):
        id = self._create_run_id()
        super().__init__(id=id, dir=runs_dir / id)
        self.cost = InferenceCost()
        self.observation_stack: list[Observation] = []
        self.store = ContextStore()

    def _create_run_id(self) -> str:
        timestamp = now().strftime("%Y%m%d_%H%M%S")
        key = petname.generate(words=3, separator="_")
        return f"{timestamp}_{key}"

    @staticmethod
    def current():
        run = _run_context.get()
        if run is None:
            raise ValueError("No active run context")
        return run

    @staticmethod
    def current_task() -> Task:
        stack = Run.current().observation_stack
        if not stack:
            raise ValueError("No active task context")
        return cast(Task, stack[-1])

    @staticmethod
    def start(runs_dir: Path) -> Run:
        run = Run(runs_dir)
        _run_context.set(run)
        return run

    @staticmethod
    def end():
        _run_context.set(None)

    @contextmanager
    def create_observation(
        self,
        name: str,
        idx: int | None = None,
    ) -> Iterator[Observation]:
        stack = self.observation_stack.copy()
        if not stack:
            raise ValueError("Observation stack unexpectedly empty")
        parent = stack[-1]
        observation = Observation(
            id=name,
            dir=parent.dir / (name + (f"_{idx}" if idx else "")),
        )
        self.observation_stack = stack + [observation]
        try:
            yield observation
        finally:
            observation.end()
            stack = self.observation_stack.copy()
            stack.pop()
            self.observation_stack = stack
