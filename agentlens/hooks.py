from inspect import signature
from typing import Any, Callable, Generator, Sequence, TypeVar, get_type_hints

T = TypeVar("T")
R = TypeVar("R")

GeneratorHook = Generator[dict[str, Any], T, None]
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
        """Subclasses should implement this to check specific return type requirements"""
        raise NotImplementedError


class Hook(Wrapper):
    """A hook that can intercept and modify function calls"""

    def __call__(self, args: tuple, kwargs: dict) -> GeneratorHook | None:
        """Execute the hook around a function call"""
        mock_kwargs = self._build_kwargs(args, kwargs)
        return self.callback(**mock_kwargs)

    def _check_return_type(self, callback_return: type, target_return: type) -> None:
        """Temporarily disabled type checking"""
        pass


class Mock(Wrapper):
    """A mock that replaces a function call"""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the mock function with validated arguments"""
        mock_kwargs = self._build_kwargs(args, kwargs)
        return self.callback(**mock_kwargs)

    def _check_return_type(self, callback_return: type, target_return: type) -> None:
        if callback_return != target_return:
            raise TypeError(
                f"Mock callback must return same type as target. "
                f"Expected {target_return}, got {callback_return}"
            )


class MockMiss(Exception):
    """Raised by mock functions to indicate the real function should be called"""

    pass


def convert_to_kwargs(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    bound_args = signature(fn).bind(*args, **kwargs)
    bound_args.apply_defaults()
    return dict(bound_args.arguments)


# todo is this used
def initialize_hooks(
    example: T,
    hook_factories: Sequence[Callable[[T], Hook]] | None,
) -> dict[str, list[Hook]]:
    hooks: dict[str, list[Hook]] = {}
    for hook_factory in hook_factories or []:
        hook = hook_factory(example)
        target_name = hook.target.__name__
        if target_name not in hooks:
            hooks[target_name] = []
        hooks[target_name].append(hook)
    return hooks
