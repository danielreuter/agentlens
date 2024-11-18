import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def serialize_value(value: Any) -> Any:
    """Convert a Python value to a JSON-serializable format."""
    # Handle None
    if value is None:
        return None

    # Handle primitives
    if isinstance(value, (str, int, float, bool)):
        return value

    # Handle Pydantic models
    if isinstance(value, BaseModel):
        return value.model_dump()

    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]

    # Handle dicts
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}

    # Fallback: convert to string
    return str(value)


def write_json(data: dict, path: Path) -> None:
    """Write data to a JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
