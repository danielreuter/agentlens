from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

from agentlens.dataset import Row


class Hook:
    def __init__(self, cb: Callable, target: Callable, row: Row, **kwargs):
        self.cb = cb
        self.target = target
        self.row = row
        self.kwargs = kwargs

    def __call__(self, output, *args, **kwargs):
        return self.cb(self.row, output, *args, **kwargs, **self.kwargs)


class Hooks:
    _instance: ContextVar[Hooks | None] = ContextVar("hooks_instance", default=None)

    def __init__(self, hooks: list[Hook]):
        self.hooks: dict[str, list[Hook]] = {}
        for hook in hooks:
            self.register(hook)

    @classmethod
    def get(cls) -> Hooks:
        instance = cls._instance.get()
        if instance is None:
            raise RuntimeError("Hooks instance not created")
        return instance

    @classmethod
    @contextmanager
    def create(cls, hooks: list[Hook]):
        instance = cls(hooks)
        token = cls._instance.set(instance)
        try:
            yield instance
        finally:
            cls._instance.reset(token)

    def register(self, hook: Hook) -> None:
        target_name = hook.target.__name__
        if target_name not in self.hooks:
            self.hooks[target_name] = []
        self.hooks[target_name].append(hook)

    def __getitem__(self, func_name: str) -> list[Hook]:
        return self.hooks.get(func_name, [])
