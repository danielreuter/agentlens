from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from agentlens.dataset import Row
from agentlens.hooks import Hook
from agentlens.utils import now


class Observation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    error: str | None = None
    start_time: datetime = Field(default_factory=now)
    end_time: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[Observation] = Field(default_factory=list)

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

    def get_duration(self) -> str:
        if self.end_time is None:
            return "..."
        return f"{self.end_time - self.start_time:.2f}s"


class Run:
    def __init__(self, dir: Path, name: str, row: Row, hooks: dict[str, list[Hook]]):
        self.dir = dir
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
