from __future__ import annotations

import textwrap
from typing import Literal, Union

from pydantic import BaseModel


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentUrl(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageContentUrl


MessageRole = Literal["system", "user", "assistant"]

RawMessageContent = Union[str, tuple[Union[str, ImageContent], ...]]
"""Text content is passed in as a string"""

MessageContent = Union[str, list[Union[TextContent, ImageContent]]]
"""Text content has been formatted as JSON"""


class Message(BaseModel):
    """An AI chat message"""

    role: MessageRole
    content: MessageContent

    @staticmethod
    def message(role: MessageRole, raw_content: RawMessageContent) -> Message:
        content = raw_content

        # convert tuple to list and replace text strings with TextContent
        if isinstance(content, tuple):
            content = [
                TextContent(text=text) if isinstance(text, str) else text for text in content
            ]

        return Message(
            role=role,
            content=content,
        )

    @staticmethod
    def system(raw_content: RawMessageContent) -> Message:
        return Message.message("system", raw_content)

    @staticmethod
    def user(raw_content: RawMessageContent) -> Message:
        return Message.message("user", raw_content)

    @staticmethod
    def assistant(raw_content: RawMessageContent) -> Message:
        return Message.message("assistant", raw_content)

    @staticmethod
    def image(url: str) -> ImageContent:
        return ImageContent(
            type="image_url",
            image_url=ImageContentUrl(url=url),
        )

    def dedent(self) -> Message:
        if isinstance(self.content, str):
            return Message(role=self.role, content=textwrap.dedent(self.content))
        else:
            return Message(
                role=self.role,
                content=[
                    TextContent(text=textwrap.dedent(item.text))
                    if isinstance(item, TextContent)
                    else item
                    for item in self.content
                ],
            )


def user_message(raw_content: RawMessageContent) -> Message:
    return Message.user(raw_content)


def system_message(raw_content: RawMessageContent) -> Message:
    return Message.system(raw_content)


def assistant_message(raw_content: RawMessageContent) -> Message:
    return Message.assistant(raw_content)


def image_message(url: str) -> ImageContent:
    return Message.image(url)
