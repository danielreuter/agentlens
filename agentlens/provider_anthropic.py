from typing import Any, Type, TypeVar

import anthropic
from pydantic import BaseModel

from agentlens.provider import InferenceCost, Message, Provider

T = TypeVar("T", bound=BaseModel)


class Anthropic(Provider):
    # Cost per million tokens in USD
    MODEL_COSTS = {
        "claude-3-5-sonnet-20240620": (3.00, 15.00),
        "claude-3-opus-20240229": (15.00, 75.00),
        "claude-3-sonnet-20240229": (3.00, 15.00),
        "claude-3-haiku-20240307": (0.25, 1.25),
    }

    def __init__(
        self,
        api_key: str | None = None,
        max_connections: dict[str, int] | None = None,
        max_connections_default: int = 10,
    ):
        super().__init__(
            name="anthropic",
            max_connections=max_connections,
            max_connections_default=max_connections_default,
        )
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def get_token_costs(self, model: str) -> tuple[float, float]:
        """Returns (input_cost_per_million, output_cost_per_million)"""
        return self.MODEL_COSTS.get(model, (0.0, 0.0))

    def _update_cost(
        self,
        model: str,
        response: Any,
        inference_cost: InferenceCost | None,
    ) -> None:
        if inference_cost is None:
            return

        usage = getattr(response, "usage", None)
        if usage:
            input_cost_per_token, output_cost_per_token = self.get_token_costs(model)
            inference_cost.input_cost += usage.input_tokens * input_cost_per_token / 1_000_000
            inference_cost.output_cost += usage.output_tokens * output_cost_per_token / 1_000_000

    async def generate_text(
        self,
        *,
        model: str,
        messages: list[Message],
        inference_cost: InferenceCost | None = None,
        **kwargs,
    ) -> str:
        completion = await self.client.messages.create(
            model=model,
            messages=[m.model_dump() for m in messages],  # type: ignore
            **kwargs,
        )

        self._update_cost(model, completion, inference_cost)
        assert completion.content is not None
        return completion.content

    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: Type[T],
        **kwargs,
    ) -> T:
        raise NotImplementedError("Anthropic does not support object generation")
