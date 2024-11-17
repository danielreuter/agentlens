from typing import Any, Awaitable, Callable

from agentlens.hooks import Hook


def hook(target_fn: Callable[..., Awaitable[Any]]) -> Callable[[Callable], Hook]:
    def decorator(hook_fn: Callable) -> Hook:
        if not hasattr(hook_fn, "__name__"):
            raise ValueError("Hook function must have a __name__ attribute")
        return Hook(hook_fn, target_fn)

    return decorator


# todo-- add mock
