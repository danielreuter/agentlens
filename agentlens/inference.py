from __future__ import annotations

import asyncio
import logging
import random
import textwrap
from abc import ABC
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Type, TypeVar, overload

from pydantic import BaseModel
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential

from agentlens.client import observe

# Configuration constants
DEFAULT_TIMEOUT_SECONDS = 480
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_MAX_SECONDS = 300  # 5 minutes max between retries

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentUrl(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageContentUrl


MessageRole = Literal["system", "user", "assistant"]

RawMessageContent = str | ImageContent | dict[str, str | dict]
"""Text content is passed in as a string or dictionary"""

MessageContent = list[TextContent | ImageContent] | TextContent | ImageContent | str
"""Text content has been formatted as JSON"""


class Message(BaseModel):
    """An AI chat message"""

    role: MessageRole
    content: MessageContent

    @staticmethod
    def _format_content(
        content: RawMessageContent, dedent: bool = True
    ) -> TextContent | ImageContent:
        if isinstance(content, (str, dict)):
            text = format_prompt(content)
            return TextContent(text=textwrap.dedent(text) if dedent else text)
        else:
            return content

    @staticmethod
    def message(role: MessageRole, *raw_content: RawMessageContent, dedent: bool = True) -> Message:
        if len(raw_content) == 1:
            content = Message._format_content(raw_content[0], dedent)
            return Message(
                role=role, content=content.text if isinstance(content, TextContent) else content
            )
        else:
            return Message(
                role=role, content=[Message._format_content(item, dedent) for item in raw_content]
            )

    @staticmethod
    def system(*raw_content: RawMessageContent, dedent: bool = True) -> Message:
        return Message.message("system", *raw_content, dedent=dedent)

    @staticmethod
    def user(*raw_content: RawMessageContent, dedent: bool = True) -> Message:
        return Message.message("user", *raw_content, dedent=dedent)

    @staticmethod
    def assistant(*raw_content: RawMessageContent, dedent: bool = True) -> Message:
        return Message.message("assistant", *raw_content, dedent=dedent)

    @staticmethod
    def image_content(url: str) -> ImageContent:
        return ImageContent(
            type="image_url",
            image_url=ImageContentUrl(url=url),
        )


def user_message(*raw_content: RawMessageContent, dedent: bool = True) -> Message:
    return Message.user(*raw_content, dedent=dedent)


def system_message(*raw_content: RawMessageContent, dedent: bool = True) -> Message:
    return Message.system(*raw_content, dedent=dedent)


def assistant_message(*raw_content: RawMessageContent, dedent: bool = True) -> Message:
    return Message.assistant(*raw_content, dedent=dedent)


def image_content(url: str) -> ImageContent:
    return Message.image_content(url)


def format_prompt(prompt_input: str | dict[str, str | dict]) -> str:
    """Convert a string or nested dictionary into XML-formatted text."""
    if isinstance(prompt_input, str):
        return prompt_input

    xml_tags = []
    for key, value in prompt_input.items():
        if not value:
            continue

        if isinstance(value, dict):
            content = format_prompt(value)  # Recursively handle nested dictionaries
        else:
            content = textwrap.dedent(str(value)).strip()

        xml_tags.append(f"<{key}>\n{content}\n</{key}>")

    return "\n".join(xml_tags)


@dataclass
class Model:
    name: str
    provider: ModelProvider


class ModelProvider(ABC):
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

    async def generate_text(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        raise NotImplementedError

    async def generate_object(
        self,
        *,
        model: str,
        messages: list[Message],
        schema: Type[T],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> T:
        raise NotImplementedError

    def __truediv__(self, model: str) -> Model:
        return Model(name=model, provider=self)


class GenerationTimeoutError(Exception):
    """Raised when text generation exceeds the specified timeout."""

    pass


@observe
async def generate_text(
    model: Model,
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    capture_messages: bool = True,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
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
        capture_messages=capture_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )


@overload
async def generate_object(
    model: Model,
    schema: Type[T],
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    capture_messages: bool = True,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> T: ...


@overload
async def generate_object(
    model: Model,
    schema: dict[str, Any],
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    capture_messages: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]: ...


@observe
async def generate_object(
    model: Model,
    schema: Type[T] | dict[str, Any],
    messages: list[Message] | None = None,
    system: str | dict[str, str | dict] | None = None,
    prompt: str | dict[str, str | dict] | None = None,
    dedent: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    capture_messages: bool = False,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
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
        capture_messages=capture_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
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
    capture_messages: bool,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    **kwargs,
) -> Any:
    collected_messages = _create_messages(
        messages=messages,
        system=system,
        prompt=prompt,
        dedent=dedent,
    )
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=DEFAULT_BACKOFF_MAX_SECONDS),
            reraise=True,
        ):
            with attempt:
                try:
                    async with semaphore:
                        await asyncio.sleep(random.uniform(0, 0.1))
                        async with asyncio.timeout(timeout):
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
