from typing import Type, TypeVar

from langfuse.openai import AsyncOpenAI
from pydantic import BaseModel

from agentlens.provider import Message, Provider

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(Provider):
    def __init__(
        self,
        api_key: str | None = None,
        max_connections: dict[str, int] = {"DEFAULT": 10},
    ):
        super().__init__(name="openai", max_connections=max_connections)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate_text(
        self,
        *,
        model: str,
        messages: list[Message],
        **kwargs,
    ) -> str:
        completion = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        assert completion.choices[0].message.content is not None
        return completion.choices[0].message.content

    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        type: Type[T],
        **kwargs,
    ) -> T:
        completion = await self.client.beta.chat.completions.parse(
            model=model,
            messages=[message.model_dump() for message in messages],
            response_format=type,
            **kwargs,
        )
        assert completion.choices[0].message.parsed is not None
        return completion.choices[0].message.parsed
