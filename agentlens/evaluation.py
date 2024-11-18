from inspect import signature
from typing import (
    Any,
    Awaitable,
    Callable,
    Generator,
    TypeVar,
    get_type_hints,
)

T = TypeVar("T")
R = TypeVar("R")

HookGenerator = Generator[dict[str, Any], T, None]
"""A wrapper-type hook"""


class Wrapper:
    """Base class for function wrappers that need to validate and reconstruct arguments"""

    def __init__(self, callback: Callable, target: Callable):
        self.callback = callback
        self.target = target
        self._validate_params()
        self._validate_return_type()

    def _validate_params(self) -> None:
        """Validate that callback only requests parameters that exist in target function"""
        callback_sig = signature(self.callback)
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
        """Build the kwargs dictionary for the callback based on the target function's args"""
        callback_sig = signature(self.callback)
        callback_kwargs = {}

        # Extract only the parameters that the callback requested
        all_args = dict(zip(signature(self.target).parameters, args))
        all_args.update(kwargs)

        for param_name in callback_sig.parameters:
            if param_name in all_args:
                callback_kwargs[param_name] = all_args[param_name]

        return callback_kwargs

    def _validate_return_type(self) -> None:
        """Validate the return type of the callback matches expectations"""
        # Skip validation if type hints are missing
        target_hints = get_type_hints(self.target)
        callback_hints = get_type_hints(self.callback)

        if not (target_hints.get("return") and callback_hints.get("return")):
            return

        self._check_return_type(callback_hints["return"], target_hints["return"])

    def _check_return_type(self, callback_return: type, target_return: type) -> None:
        pass


class Hook(Wrapper):
    """A hook that can intercept and modify function calls"""

    def __call__(self, args: tuple, kwargs: dict) -> HookGenerator | None:
        """Execute the hook around a function call"""
        mock_kwargs = self._build_kwargs(args, kwargs)
        return self.callback(**mock_kwargs)


class Mock(Wrapper):
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


def convert_to_kwargs(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    bound_args = signature(fn).bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)


def hook(target_fn: Callable[..., Awaitable[Any]]) -> Callable[[Callable], Hook]:
    def decorator(hook_fn: Callable) -> Hook:
        if not hasattr(hook_fn, "__name__"):
            raise ValueError("Hook function must have a __name__ attribute")
        return Hook(hook_fn, target_fn)

    return decorator


def mock(target_fn: Callable[..., Awaitable[Any]]) -> Callable[[Callable], Mock]:
    def decorator(mock_fn: Callable) -> Mock:
        if not hasattr(mock_fn, "__name__"):
            raise ValueError("Mock function must have a __name__ attribute")
        return Mock(mock_fn, target_fn)

    return decorator
