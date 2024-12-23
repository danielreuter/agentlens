from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


@dataclass
class ContextStack(Generic[T]):
    """Stack-based context manager for thread-local state"""

    name: str
    _stack: ContextVar[list[dict[str, T]]]

    def __init__(self, name: str):
        self.name = name
        self._stack = ContextVar(name, default=[])

    @property
    def current(self) -> dict[str, T] | None:
        """Get current top of stack"""
        stack = self.stack
        return stack[-1] if stack else None

    @property
    def stack(self) -> list[dict[str, T]]:
        return self._stack.get()

    @contextmanager
    def push(self, new_map: dict[str, T]) -> Iterator[dict[str, T]]:
        token = self._stack.set(self.stack + [new_map])
        try:
            yield new_map
        finally:
            self._stack.reset(token)

    def use(self, named_object: Any) -> T:
        name = get_cls_name_or_raise(named_object)
        current = self.current
        if not current or name not in current:
            raise ValueError(f"Context {name} not found")
        return current[name]


def get_cls_name_or_raise(cls: type[BaseModel]) -> str:
    """Get the class name for a Pydantic BaseModel class."""
    if not hasattr(cls, "__name__"):
        raise ValueError(f"Class {cls} has no __name__ attribute")
    return cls.__name__


def get_fn_name_or_raise(fn: Callable) -> str:
    if not (name := getattr(fn, "__name__", None)):
        raise ValueError(
            f"Function {fn} has no __name__ attribute. Please provide an object with a __name__ attribute"
        )
    return name
