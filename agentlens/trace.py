from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentlens.dataset import Row
from agentlens.hooks import Hook
from agentlens.utils import now

T = TypeVar("T", bound=Row)


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
    def __init__(self, key: str, dir: Path, name: str, row: T, hooks: dict[str, list[Hook]]):
        self.key = key
        self.dir = dir.resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hooks = hooks
        self.row = row
        self.observation = Observation(name=name)
        self.observation_stack: list[Observation] = [self.observation]


class Generation(Observation):
    model: str
    prompt_tokens: int
    output_tokens: int
    cost: float
