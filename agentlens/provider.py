from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from textwrap import dedent
from typing import Type, TypeVar

from agentlens.message import Message

T = TypeVar("T")


@dataclass
class InferenceCost:
    input: float = 0.0
    output: float = 0.0

    @property
    def total(self) -> int:
        return self.input + self.output

    def __str__(self) -> str:
        return dedent(
            f"""\
            Inference cost:
            - Input cost: ${self.input:.6f}
            - Output cost: ${self.output:.6f}
            - Total cost: ${self.total:.6f}
            """
        )


class Provider(ABC):
    def __init__(
        self,
        name: str,
        max_connections: dict[str, int] = {"DEFAULT": 10},
    ):
        self.name = name
        self._semaphores: dict[str, asyncio.Semaphore] = {}

        for model, limit in max_connections.items():
            self._semaphores[model] = asyncio.Semaphore(limit)

        self._default_semaphore = asyncio.Semaphore(max_connections.get("DEFAULT", 10))

    def get_semaphore(self, model: str) -> asyncio.Semaphore:
        return self._semaphores.get(model, self._default_semaphore)

    @abstractmethod
    async def generate_text(
        self,
        *,
        model: str,
        messages: list[Message],
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: Type[T],
        **kwargs,
    ) -> T:
        pass
