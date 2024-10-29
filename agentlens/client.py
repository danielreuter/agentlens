import asyncio
import logging
import random
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, ParamSpec, Sequence, Type, TypeVar

import nest_asyncio
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from pydantic import BaseModel
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from agentlens.cache import TaskCache
from agentlens.dataset import Dataset, Row
from agentlens.hooks import Hook, Hooks
from agentlens.log import create_run_log
from agentlens.provider import Message, Provider
from agentlens.serialization import serialize_task_input, serialize_task_output
from agentlens.utils import create_path

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
R = TypeVar("R")
RunT = TypeVar("RunT")


class AI:
    langfuse: Langfuse | None
    message: Type[Message] = Message
    _dataset_dir: Path
    _run_dir: Path
    _providers: dict[str, Provider]

    def __init__(
        self,
        *,
        dataset_dir: Path | str,
        run_dir: Path | str,
        langfuse: Langfuse | None = None,
        providers: Sequence[Provider] = (),
    ):
        self.langfuse = langfuse
        self._dataset_dir = create_path(dataset_dir)
        self._run_dir = create_path(run_dir)
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
        type: Type[T],
        messages: list[Message] | None = None,
        system: str | None = None,
        prompt: str | None = None,
        dedent: bool = True,
        max_retries: int = 3,
        **kwargs,
    ) -> T:
        # inline types may have invalid __name__ attributes -- replace w/ default
        if hasattr(type, "__name__"):
            type.__name__ = "Response"
        provider, model_name = self._get_provider(model)
        return await self._generate(
            provider.generate_object,
            model=model_name,
            semaphore=provider.get_semaphore(model_name),
            type=type,
            messages=messages,
            system=system,
            prompt=prompt,
            dedent=dedent,
            max_retries=max_retries,
            **kwargs,
        )

    def run(
        self,
        *,
        main: Callable[[Any], Awaitable[RunT]],
        hooks: list[Callable[[Row], Hook]] | None = None,
        dataset: Dataset,
    ) -> list[RunT]:
        # sketchy -- should only be used in evals
        nest_asyncio.apply()

        async def run_with_hooks(row: Row):
            _hooks = [hook_factory(row) for hook_factory in (hooks or [])]
            with Hooks.create(_hooks):
                return await main(row)

        @observe()
        async def run_all():
            url = langfuse_context.get_current_trace_url()
            if url:
                print(f"View trace: {url}")
            tasks = [run_with_hooks(row) for row in dataset]
            return await asyncio.gather(*tasks)

        with TaskCache.enable(self._dataset_dir / "cache"):
            with create_run_log(self._run_dir) as log:
                return asyncio.run(run_all(langfuse_observation_id=log.run_id))

    def hook(self, target_func: Callable, **kwargs) -> Callable[[Callable], Callable[[Row], Hook]]:
        def decorator(cb: Callable) -> Callable[[Row], Hook]:
            @wraps(cb)
            def wrapper(row: Row) -> Hook:
                return Hook(cb, target_func, row, **kwargs)

            return wrapper

        return decorator

    def score(
        self,
        name: str,
        value: Any,
        comment: str | None = None,
    ):
        if self.langfuse:
            self.langfuse.score(
                name=name,
                value=value,
                trace_id=langfuse_context.get_current_trace_id(),
                observation_id=langfuse_context.get_current_observation_id(),
                comment=comment,
            )

    def task(
        self,
        cache: bool = False,
        capture_input: bool = True,
        capture_output: bool = True,
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
        def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
            task_name = func.__name__

            # conditionally cache
            if cache:
                func = TaskCache.cached(func)

            @wraps(func)
            @observe(capture_input=capture_input, capture_output=capture_output)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # execute function
                output = await func(*args, **kwargs)

                # run any hooks
                if hooks := Hooks.get():
                    for hook in hooks[task_name]:
                        hook(output, *args, **kwargs)

                # log to Langfuse
                if self.langfuse:
                    langfuse_context.update_current_observation(
                        name=task_name,
                        input=serialize_task_input(args, kwargs) if capture_input else None,
                        output=serialize_task_output(output) if capture_output else None,
                    )

                return output

            return wrapper

        return decorator

    def dataset(self, name: str) -> Callable[[Type[D]], Type[D]]:
        def decorator(cls: Type[D]) -> Type[D]:
            cls.name = name
            cls.dataset_dir = self._dataset_dir
            return cls

        return decorator
