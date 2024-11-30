from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from textwrap import dedent
from typing import Type, TypeVar, overload

from pydantic import BaseModel

from agentlens.message import Message

T = TypeVar("T", bound=BaseModel)


@dataclass
class InferenceCost:
    input: float = 0.0
    output: float = 0.0

    @property
    def total(self) -> float:
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


@dataclass
class Model:
    name: str
    provider: Provider


class Provider(ABC):
    def __init__(
        self,
        name: str,
        max_connections: dict[str, int] | None = None,
        max_connections_default: int = 10,
    ):
        self.name = name
        self._semaphores: dict[str, asyncio.Semaphore] = {}

        if max_connections is not None:
            for model, limit in max_connections.items():
                self._semaphores[model] = asyncio.Semaphore(limit)

        self._default_semaphore = asyncio.Semaphore(max_connections_default)

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

    @overload
    @abstractmethod
    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: Type[T],
        **kwargs,
    ) -> T: ...

    @overload
    @abstractmethod
    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: dict,
        **kwargs,
    ) -> dict: ...

    @abstractmethod
    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: Type[T] | dict,
        **kwargs,
    ) -> T | dict:
        pass

    def __truediv__(self, model: str) -> Model:
        return Model(name=model, provider=self)
