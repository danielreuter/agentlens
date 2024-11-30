import json
from typing import Any, Type, TypeVar, overload

from openai import AsyncOpenAI
from pydantic import BaseModel

from agentlens.client import lens
from agentlens.model import ModelUsage
from agentlens.provider import Message, Provider

T = TypeVar("T", bound=BaseModel)


class OpenAI(Provider):
    # Cost per million tokens in USD
    MODEL_COSTS = {
        "gpt-4o-mini": (0.150, 0.600),
        "gpt-4o-mini-2024-07-18": (0.150, 0.600),
        "gpt-4o": (5.00, 15.00),
        "gpt-4o-2024-08-06": (2.50, 10.00),
        "gpt-4o-2024-05-13": (5.00, 15.00),
        "o1-mini": (3.00, 12.00),
        "o1-mini-2024-09-12": (3.00, 12.00),
        "o1-preview": (15.00, 60.00),
        "o1-preview-2024-09-12": (15.00, 60.00),
    }

    def __init__(
        self,
        api_key: str | None = None,
        max_connections: dict[str, int] | None = None,
        max_connections_default: int = 10,
    ):
        super().__init__(
            name="openai",
            max_connections=max_connections,
            max_connections_default=max_connections_default,
        )
        self.client = AsyncOpenAI(api_key=api_key)

    def get_token_costs(self, model: str) -> tuple[float, float]:
        """Returns (input_cost_per_million, output_cost_per_million)"""
        return self.MODEL_COSTS.get(model, (0.0, 0.0))

    def _extract_usage(self, response: Any) -> ModelUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data found in OpenAI response")
        return ModelUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def _update_cost(
        self,
        model: str,
        response: Any,
    ) -> None:
        usage = self._extract_usage(response)
        input_cost_per_token, output_cost_per_token = self.get_token_costs(model)
        lens.increase_cost(
            input=usage.input_tokens * input_cost_per_token / 1_000_000,
            output=usage.output_tokens * output_cost_per_token / 1_000_000,
        )

    async def generate_text(
        self,
        *,
        model: str,
        messages: list[Message],
        **kwargs,
    ) -> str:
        completion = await self.client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in messages],  # type: ignore
            **kwargs,
        )

        self._update_cost(model, completion)
        assert completion.choices[0].message.content is not None
        return completion.choices[0].message.content

    @overload
    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: Type[T],
        **kwargs,
    ) -> T: ...

    @overload
    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: dict,
        **kwargs,
    ) -> dict: ...

    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: Type[T] | dict,
        **kwargs,
    ) -> T | dict:
        if not isinstance(schema, dict):
            completion = await self.client.beta.chat.completions.parse(
                model=model,
                messages=[message.model_dump() for message in messages],  # type: ignore
                response_format=schema,
                **kwargs,
            )

            self._update_cost(model, completion)
            assert completion.choices[0].message.parsed is not None
            return completion.choices[0].message.parsed
        else:
            completion = await self.client.chat.completions.create(
                model=model,
                messages=[m.model_dump() for m in messages],  # type: ignore
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "params",
                        "schema": schema,
                        "strict": True,
                    },
                },
                **kwargs,
            )

            self._update_cost(model, completion)
            assert completion.choices[0].message.content is not None
            return json.loads(completion.choices[0].message.content)
