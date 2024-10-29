from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeVar,
)

import petname
from openai.types import CompletionUsage as OpenAIUsage
from openai.types.chat import (
    ChatCompletion,
)

if TYPE_CHECKING:
    from anthropic.types import Usage as AnthropicUsage

logger = logging.getLogger(__name__)


def create_id():
    return str(uuid.uuid4())


def create_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def create_readable_id() -> str:
    return petname.generate(words=4, separator="_")


def join_with_dashes(*args) -> str:
    return "-".join(args)


def merge_args(*args, **kwargs) -> dict:
    return {"args": args, "kwargs": kwargs}


T_Model = TypeVar("T_Model", bound="Response")


class Response(Protocol):
    usage: OpenAIUsage | AnthropicUsage


def update_total_usage(
    response: T_Model | None,
    total_usage: OpenAIUsage | AnthropicUsage,
) -> T_Model | ChatCompletion | None:
    if response is None:
        return None

    response_usage = getattr(response, "usage", None)
    if isinstance(response_usage, OpenAIUsage) and isinstance(total_usage, OpenAIUsage):
        total_usage.completion_tokens += response_usage.completion_tokens or 0
        total_usage.prompt_tokens += response_usage.prompt_tokens or 0
        total_usage.total_tokens += response_usage.total_tokens or 0
        response.usage = total_usage  # Replace each response usage with the total usage
        return response

    # Anthropic usage.
    try:
        from anthropic.types import Usage as AnthropicUsage

        if isinstance(response_usage, AnthropicUsage) and isinstance(total_usage, AnthropicUsage):
            total_usage.input_tokens += response_usage.input_tokens or 0
            total_usage.output_tokens += response_usage.output_tokens or 0
            response.usage = total_usage
            return response
    except ImportError:
        pass

    logger.debug("No compatible response.usage found, token usage not updated.")
    return response


def disable_pydantic_error_url():
    os.environ["PYDANTIC_ERRORS_INCLUDE_URL"] = "0"
