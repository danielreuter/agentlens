import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Type, TypeVar

from agentlens.message import Message

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
        dedent: bool,
        max_retries: int,
        **kwargs,
    ) -> str:
        pass

    @abstractmethod
    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        type: Type[T],
        **kwargs,
    ) -> T:
        pass
