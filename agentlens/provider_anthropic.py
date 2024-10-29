from typing import Type, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from agentlens.provider import Message, Provider

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(Provider):
    def __init__(
        self,
        api_key: str | None = None,
        max_connections: dict[str, int] = {"DEFAULT": 10},
    ):
        super().__init__(name="anthropic", max_connections=max_connections)
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate_text(
        self,
        *,
        model: str,
        messages: list[Message],
        **kwargs,
    ) -> str:
        completion = await self.client.messages.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        assert completion.content is not None
        return completion.content

    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        type: Type[T],
        **kwargs,
    ) -> T:
        raise NotImplementedError("Anthropic does not support object generation")
