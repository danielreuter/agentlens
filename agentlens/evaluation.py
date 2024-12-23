from inspect import Parameter, signature
from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    TypeVar,
)

T = TypeVar("T")
R = TypeVar("R")

Hook = Generator[dict[str, Any] | None, T, None]
"""A wrapper-type hook"""

GLOBAL_HOOK_KEY = "__global__"  # NEW: Special key for global hooks


class Wrapper:
    """Base class for function wrappers that need to validate and reconstruct arguments"""

    def __init__(self, callback: Callable, target: Callable | None):
        self.callback = callback
        self.target = target
        self._validate_params()

    def _validate_params(self) -> None:
        """Skip strict param-check if callback uses *args or **kwargs, or if target is None"""
        if self.target is None:
            return

        callback_sig = signature(self.callback)
        # Skip validation if callback uses *args or **kwargs
        if any(
            p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
            for p in callback_sig.parameters.values()
        ):
            return

        # Normal validation for specific parameters
        target_sig = signature(self.target)
        callback_params = callback_sig.parameters
        target_params = target_sig.parameters

        for name in callback_params:
            if name not in target_params:
                raise ValueError(
                    f"Parameter '{name}' does not exist in target function {self.target.__name__}. "
                    f"Valid parameters are: {list(target_params.keys())}"
                )

    def _build_kwargs(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """Build the kwargs dictionary for the callback based on its signature"""
        callback_sig = signature(self.callback)

        # If callback uses *args or **kwargs, pass everything
        if any(
            p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
            for p in callback_sig.parameters.values()
        ):
            return {**{str(i): v for i, v in enumerate(args)}, **kwargs}

        # Otherwise, filter by param name
        if self.target is not None:
            target_sig = signature(self.target)
            all_args = dict(zip(target_sig.parameters, args))
        else:
            all_args = {str(i): v for i, v in enumerate(args)}
        all_args.update(kwargs)

        callback_kwargs = {}
        for param_name in callback_sig.parameters:
            if param_name in all_args:
                callback_kwargs[param_name] = all_args[param_name]

        return callback_kwargs


class HookFn(Wrapper):
    """A hook that can intercept and modify function calls"""

    def __call__(self, args: tuple, kwargs: dict) -> Hook | None:
        """Execute the hook around a function call"""
        mock_kwargs = self._build_kwargs(args, kwargs)
        return self.callback(**mock_kwargs)


class MockFn(Wrapper):
    """A mock that replaces a function call"""

    target_name: str  # Store the target function name for lookup

    def __init__(self, callback: Callable, target: Callable):
        super().__init__(callback, target)
        self.target_name = target.__name__

    async def __call__(self, **kwargs: Any) -> Any:
        """Execute the mock function with validated arguments"""
        # Filter kwargs to only those the mock accepts
        mock_kwargs = self._build_kwargs((), kwargs)
        result = await self.callback(**mock_kwargs)
        return result


class MockMiss(Exception):
    """Raised by mock functions to indicate the real function should be called"""

    pass


def format_input_dict(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    bound_args = signature(fn).bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)


def hook(
    target_fn: Callable[..., Awaitable[Any]] | None = None,
) -> Callable[[Callable], HookFn]:  # CHANGED: Made target_fn optional
    def decorator(hook_fn: Callable) -> HookFn:
        if not hasattr(hook_fn, "__name__"):
            raise ValueError("Hooked functions must have a __name__ attribute")
        return HookFn(hook_fn, target_fn)

    if callable(target_fn):
        return decorator
    return decorator


def mock(target_fn: Callable[..., Awaitable[Any]]) -> Callable[[Callable], MockFn]:
    def decorator(mock_fn: Callable) -> MockFn:
        if not hasattr(mock_fn, "__name__"):
            raise ValueError("Mocked functions must have a __name__ attribute")
        return MockFn(mock_fn, target_fn)

    return decorator
