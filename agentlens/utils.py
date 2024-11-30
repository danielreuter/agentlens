from __future__ import annotations

import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now() -> datetime:
    return datetime.now(timezone.utc)


def create_uuid() -> str:
    return str(uuid.uuid4())


def create_path(path: Path | str) -> Path:
    return path if isinstance(path, Path) else Path(path)


def join_with_dashes(*args) -> str:
    return "-".join(args)


def merge_args(*args, **kwargs) -> dict:
    return {"args": args, "kwargs": kwargs}


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
