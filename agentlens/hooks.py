from __future__ import annotations

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
