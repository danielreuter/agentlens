from __future__ import annotations

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
