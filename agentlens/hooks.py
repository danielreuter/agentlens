from __future__ import annotations

from typing import Callable, Literal

from agentlens.dataset import Example


class Hook:
    def __init__(
        self,
        cb: Callable,
        target: Callable,
        row: Example,
        type: Literal["pre", "post"] = "post",
        **kwargs,
    ):
        self.cb = cb
        self.target = target
        self.row = row
        self.type = type
        self.kwargs = kwargs

    def __call__(self, state, *args, **kwargs):
        if self.type == "pre":
            # Pre hooks receive (example, state, *args, **kwargs)
            return self.cb(self.row, state, *args, **kwargs, **self.kwargs)
        else:
            # Post hooks receive (example, state, output, *args, **kwargs)
            output = args[0] if args else None
            return self.cb(self.row, state, output, *args[1:], **kwargs, **self.kwargs)
