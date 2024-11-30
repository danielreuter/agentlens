from __future__ import annotations

import textwrap
from typing import Literal

from pydantic import BaseModel

from agentlens.utils import format_prompt


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

MessageContent = list[TextContent | ImageContent] | ImageContent | str
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
            content = [Message._format_content(item, dedent) for item in raw_content]
            return Message(role=role, content=content)

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
    def image(url: str) -> ImageContent:
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
    return Message.image(url)
