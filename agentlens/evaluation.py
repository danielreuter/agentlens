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

GLOBAL_HOOK_KEY = "__global__"


class Wrapper:
    """Base class for function wrappers that need to validate and reconstruct arguments"""

    def __init__(self, callback: Callable, target: Callable | None):
        self.callback = callback
        self.target = target
        self._validate_params()

    def _validate_params(self) -> None:
        """
        Validation rules:
        1) Target function cannot have a parameter named 'input'
        2) If callback has 'input' param, it must be the only param
        3) Otherwise fall back to normal validation (unless using *args/**kwargs)
        """
        if self.target is None:
            return  # skip validation for global hooks

        # 1) Disallow 'input' in the real function's signature
        target_sig = signature(self.target)
        if "input" in target_sig.parameters:
            raise ValueError(
                f"Target function {self.target.__name__}() has a parameter named 'input'; "
                f"'input' is reserved and not allowed."
            )

        callback_sig = signature(self.callback)
        callback_params = callback_sig.parameters

        # 2) If using 'input', it must be the only parameter
        if "input" in callback_params:
            if len(callback_params) != 1:
                raise ValueError(
                    "If a hook/mock function declares a parameter named 'input', it cannot have "
                    "any additional parameters."
                )
            return

        # 3a) Skip validation if using *args/**kwargs
        if any(
            p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
            for p in callback_params.values()
        ):
            return

        # 3b) Normal parameter validation
        for name in callback_params:
            if name not in target_sig.parameters:
                raise ValueError(
                    f"Parameter '{name}' does not exist in target function {self.target.__name__}. "
                    f"Valid parameters are: {list(target_sig.parameters.keys())}"
                )

    def _build_kwargs(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        """
        Build kwargs for callback:
        - If callback has 'input' param, pass all args in a single dict
        - Otherwise use normal param matching or *args/**kwargs handling
        """
        callback_sig = signature(self.callback)
        callback_params = callback_sig.parameters

        # Get all arguments from target function if it exists
        if self.target is not None:
            target_sig = signature(self.target)
            bound_args = target_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            all_args = dict(bound_args.arguments)
        else:
            # For global hooks, combine raw args and kwargs
            all_args = {**{str(i): v for i, v in enumerate(args)}, **kwargs}

        # Special handling for 'input' parameter
        if "input" in callback_params:
            return {"input": all_args}

        # Pass everything for *args/**kwargs
        if any(
            p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
            for p in callback_params.values()
        ):
            return {**{str(i): v for i, v in enumerate(args)}, **kwargs}

        # Normal parameter matching
        callback_kwargs = {}
        for param_name in callback_params:
            if param_name in all_args:
                callback_kwargs[param_name] = all_args[param_name]
        return callback_kwargs


class HookFn(Wrapper):
    """A hook that can intercept and modify function calls"""

    def __call__(self, args: tuple, kwargs: dict) -> Hook | None:
        """Execute the hook around a function call"""
        hook_kwargs = self._build_kwargs(args, kwargs)
        return self.callback(**hook_kwargs)


class MockFn(Wrapper):
    """A mock that replaces a function call"""

    target_name: str  # Store the target function name for lookup

    def __init__(self, callback: Callable, target: Callable):
        super().__init__(callback, target)
        self.target_name = target.__name__

    async def __call__(self, **kwargs: Any) -> Any:
        """Execute the mock function with validated arguments"""
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
) -> Callable[[Callable], HookFn]:
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
