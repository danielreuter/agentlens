from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Generic, Iterator, List, TypeVar

T = TypeVar("T")


@dataclass
class ContextStack(Generic[T]):
    """Stack-based context manager for thread-local state"""

    name: str
    _stack: ContextVar[List[T]]

    def __init__(self, name: str, initial: List[T] = []):
        self.name = name
        self._stack = ContextVar(name, default=initial)

    @property
    def current(self) -> T | None:
        """Get current top of stack"""
        stack = self.stack
        return stack[-1] if stack else None

    @property
    def stack(self) -> List[T]:
        return self._stack.get()

    @contextmanager
    def push(self, item: T) -> Iterator[T]:
        token = self._stack.set(self.stack + [item])
        try:
            yield item
        finally:
            self._stack.reset(token)
