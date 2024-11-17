import asyncio
import logging
import random
from typing import Any, Callable, ParamSpec, Sequence, Type, TypeVar

from pydantic import BaseModel
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from agentlens.dataset import Dataset
from agentlens.lens import Lens
from agentlens.provider import Message, Provider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
R = TypeVar("R")
RunT = TypeVar("RunT")


class AI:
    message: Type[Message] = Message
    _providers: dict[str, Provider]
    # todo - need mechanism for spawning generations under the run
    # lens: Lens

    def __init__(
        self,
        *,
        providers: Sequence[Provider] = (),
    ):
        self._providers = {provider.name: provider for provider in providers}

    def _create_messages(
        self,
        *,
        messages: list[Message] | None = None,
        system: str | None = None,
        prompt: str | None = None,
        dedent: bool = True,
    ) -> list[Message]:
        # check for invalid combinations
        if messages and (system or prompt):
            raise ValueError("Cannot specify both 'messages' and 'system'/'prompt'")

        # create messages if passed prompts
        if not messages:
            messages = []
            if system:
                messages.append(Message.system(system))
            if prompt:
                messages.append(Message.user(prompt))

        # apply dedent if needed
        return messages if not dedent else [m.dedent() for m in messages]

    def _parse_model_id(self, model_id: str) -> tuple[str, str]:
        """Transforms 'provider:model' -> (provider, model)"""
        try:
            provider_name, model_name = model_id.split(":", 1)
            return provider_name, model_name
        except ValueError:
            raise ValueError(
                f"Invalid model identifier '{model_id}'. " f"Expected format: 'provider:model'"
            )

    def _get_provider(self, model_id: str) -> tuple[Provider, str]:
        """Get the provider and parsed model name for a given model identifier"""
        provider_name, model_name = self._parse_model_id(model_id)

        provider = self._providers.get(provider_name)
        if not provider:
            raise ValueError(
                f"Provider '{provider_name}' not configured in AI manager. "
                f"Available providers: {list(self._providers.keys())}"
            )

        return provider, model_name

    async def _generate(
        self,
        generator: Callable,
        *,
        model: str,
        semaphore: asyncio.Semaphore,
        messages: list[Message] | None,
        system: str | None,
        prompt: str | None,
        dedent: bool,
        max_retries: int,
        **kwargs,
    ) -> Any:
        collected_messages = self._create_messages(
            messages=messages,
            system=system,
            prompt=prompt,
            dedent=dedent,
        )

        # Get the current run's inference cost if available
        inference_cost = None
        try:
            run = Lens._get_current_run()
            inference_cost = run.inference_cost
        except ValueError:
            pass

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                reraise=True,
            ):
                with attempt:
                    try:
                        async with semaphore:
                            await asyncio.sleep(random.uniform(0, 0.1))
                            return await generator(
                                model=model,
                                messages=collected_messages,
                                inference_cost=inference_cost,
                                **kwargs,
                            )
                    except Exception as e:
                        logger.debug(
                            f"Retry ({attempt.retry_state.attempt_number} of {max_retries}): {e}"
                        )
                        raise e
        except RetryError as e:
            logger.debug(f"Failed after {max_retries} attempts: {e}")
            raise e

    async def generate_text(
        self,
        *,
        model: str,
        messages: list[Message] | None = None,
        system: str | None = None,
        prompt: str | None = None,
        dedent: bool = True,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        provider, model_name = self._get_provider(model)
        return await self._generate(
            provider.generate_text,
            model=model_name,
            semaphore=provider.get_semaphore(model_name),
            messages=messages,
            system=system,
            prompt=prompt,
            dedent=dedent,
            max_retries=max_retries,
            **kwargs,
        )

    async def generate_object(
        self,
        *,
        model: str,
        schema: Type[T],
        messages: list[Message] | None = None,
        system: str | None = None,
        prompt: str | None = None,
        dedent: bool = True,
        max_retries: int = 3,
        **kwargs,
    ) -> T:
        # inline types may have invalid __name__ attributes -- replace w/ a default
        if hasattr(schema, "__name__"):
            schema.__name__ = "Response"
        provider, model_name = self._get_provider(model)
        return await self._generate(
            provider.generate_object,
            model=model_name,
            semaphore=provider.get_semaphore(model_name),
            schema=schema,
            messages=messages,
            system=system,
            prompt=prompt,
            dedent=dedent,
            max_retries=max_retries,
            **kwargs,
        )
