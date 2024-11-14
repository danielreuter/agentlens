from __future__ import annotations

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Literal, Sequence, TypeVar
from uuid import UUID, uuid4

import petname
from pydantic import BaseModel, Field

from agentlens.hooks import Hook
from agentlens.utils import now

T = TypeVar("T")

# At module level
_do_mock: ContextVar[bool] = ContextVar("do_mock", default=False)
_do_cache: ContextVar[bool] = ContextVar("do_cache", default=False)


class Log(BaseModel):
    message: str
    timestamp: datetime = Field(default_factory=now)


class File(BaseModel):
    name: str
    content: str
    timestamp: datetime = Field(default_factory=now)


class Observation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    error: str | None = None
    start_time: datetime = Field(default_factory=now)
    end_time: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[Observation] = Field(default_factory=list)
    logs: list[Log] = Field(default_factory=list)
    files: list[File] = Field(default_factory=list)

    def add_child(self, child: Observation) -> None:
        self.children.append(child)

    def end(self) -> None:
        self.end_time = now()

    def get_status(self) -> Literal["running", "completed", "failed"]:
        if self.error is not None:
            return "failed"
        if self.end_time is None:
            return "running"
        return "completed"

    def get_status_icon(self) -> Any:
        return {
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
        }[self.get_status()]

    def add_log(self, message: str) -> None:
        self.logs.append(Log(message=message))

    def add_file(self, name: str, content: str) -> None:
        self.files.append(File(name=name, content=content))


class Run(Generic[T]):
    def __init__(self, runs_dir: Path, name: str):
        self.key = self._create_run_key()
        self.dir = runs_dir / self.key
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hooks: dict[str, list[Hook]] = {}
        self.observation = Observation(name=name)
        self.observation_stack: list[Observation] = [self.observation]

        # Context stores
        self.context_values: dict[str, Any] = {}

        # Initialize contextvars with default values
        self._mock_token = _do_mock.set(False)
        self._cache_token = _do_cache.set(False)

    @property
    def do_mock(self) -> bool:
        return _do_mock.get()

    @do_mock.setter
    def do_mock(self, value: bool) -> None:
        _do_mock.set(value)

    @property
    def do_cache(self) -> bool:
        return _do_cache.get()

    @do_cache.setter
    def do_cache(self, value: bool) -> None:
        _do_cache.set(value)

    def _create_run_key(self) -> str:
        timestamp = now().strftime("%Y%m%d_%H%M%S")
        id = petname.generate(words=3, separator="_")
        return f"{timestamp}_{id}"

    @contextmanager
    def create_observation(
        self,
        name: str,
        is_method: bool = False,
        func_args: tuple = (),
        func_kwargs: dict = {},
    ) -> Iterator[Observation]:
        stack = self.observation_stack.copy()
        if not stack:
            raise ValueError("Observation stack unexpectedly empty")
        parent = stack[-1]
        observation = Observation(name=name)
        parent.add_child(observation)
        self.observation_stack = stack + [observation]
        try:
            yield observation
        finally:
            observation.end()
            stack = self.observation_stack.copy()
            stack.pop()
            self.observation_stack = stack

    def initialize_hooks(
        self,
        example: T,
        hook_factories: Sequence[Callable[[T], Hook]] | None,
    ) -> dict[str, list[Hook]]:
        hooks: dict[str, list[Hook]] = {}
        for hook_factory in hook_factories or []:
            hook = hook_factory(example)
            target_name = hook.target.__name__
            if target_name not in hooks:
                hooks[target_name] = []
            hooks[target_name].append(hook)
        return hooks

    @staticmethod
    def _is_method(func: Callable) -> bool:
        params = inspect.signature(func).parameters
        return "self" in params or "cls" in params


class Generation(Observation):
    model: str
    prompt_tokens: int
    output_tokens: int
    cost: float
