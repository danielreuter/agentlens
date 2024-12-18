import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Type, TypeVar, overload

from pydantic import BaseModel
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from agentlens.client import lens
from agentlens.message import TextContent, system_message, user_message
from agentlens.provider import Message, Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def generate_text(
    model: Model,
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
    max_retries: int = 3,
    log_messages: bool = False,
    **kwargs,
) -> str:
    return await _generate(
        model.provider.generate_text,
        semaphore=model.provider.get_semaphore(model.name),
        model_name=model.name,
        messages=messages,
        system=system,
        prompt=prompt,
        dedent=dedent,
        max_retries=max_retries,
        log_messages=log_messages,
        **kwargs,
    )


@overload
async def generate_object(
    model: Model,
    schema: Type[T],
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
    max_retries: int = 3,
    log_messages: bool = False,
    **kwargs,
) -> T: ...


@overload
async def generate_object(
    model: Model,
    schema: dict[str, Any],
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
    max_retries: int = 3,
    log_messages: bool = False,
    **kwargs,
) -> dict[str, Any]: ...


async def generate_object(
    model: Model,
    schema: Type[T] | dict[str, Any],
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
    max_retries: int = 3,
    log_messages: bool = False,
    **kwargs,
) -> T | dict[str, Any]:
    if isinstance(schema, type) and hasattr(schema, "__name__"):
        schema.__name__ = "Response"
    return await _generate(
        model.provider.generate_object,
        semaphore=model.provider.get_semaphore(model.name),
        model_name=model.name,
        schema=schema,
        messages=messages,
        system=system,
        prompt=prompt,
        dedent=dedent,
        max_retries=max_retries,
        log_messages=log_messages,
        **kwargs,
    )


async def _generate(
    generate: Callable[..., Awaitable[Any]],
    semaphore: asyncio.Semaphore,
    model_name: str,
    messages: list[Message] | None,
    system: str | dict[str, str | dict] | None,
    prompt: str | dict[str, str | dict] | None,
    dedent: bool,
    max_retries: int,
    log_messages: bool,
    **kwargs,
) -> Any:
    collected_messages = _create_messages(
        messages=messages,
        system=system,
        prompt=prompt,
        dedent=dedent,
    )
    if log_messages:
        with (lens.task.dir / "messages.md").open("a") as f:
            for m in collected_messages:
                role = m.role.upper()
                content = (
                    m.content
                    if isinstance(m.content, str)
                    else "\n".join(item.text for item in m.content if isinstance(item, TextContent))
                )
                f.write(f"## {role}\n{content}\n\n")
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
                        return await generate(
                            model=model_name,
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


def _create_messages(
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
) -> list[Message]:
    # check for invalid combinations
    if messages and (system or prompt):
        raise ValueError("Cannot specify both 'messages' and 'system'/'prompt'")

    # create messages if passed prompts
    if not messages:
        messages = []
        if system:
            messages.append(system_message(system, dedent=dedent))
        if prompt:
            messages.append(user_message(prompt, dedent=dedent))

    return messages
