from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    Coroutine,
    Generator,
    Literal,
    ParamSpec,
    TypeVar,
    overload,
)
from uuid import UUID, uuid4

from agentlens.context import ContextStack, get_cls_name_or_raise, get_fn_name_or_raise
from agentlens.evaluation import (
    Hook,
    HookFn,
    MockFn,
    MockMiss,
    format_input_dict,
)

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R", covariant=True)
F = TypeVar("F", bound=Callable[..., Any])


_hooks = ContextStack[list[HookFn]]("hooks")  # fn_name -> list[Hooks]
_mocks = ContextStack[MockFn]("mocks")  # fn_name -> Mock
_contexts = ContextStack[Any]("contexts")  # cls_name -> context

ObjectT = TypeVar("ObjectT")


def use(named_object: type[ObjectT]) -> ObjectT:
    return _contexts.use(named_object)


@dataclass
class Observation:
    id: UUID
    parent: Observation | None
    children: list[Observation]


@contextmanager
def provide(
    *contexts: Any,
    hooks: list[HookFn] = [],
    mocks: list[MockFn] = [],
    on_conflict: Literal["raise", "nest"] = "raise",
) -> Generator[None, None, None]:
    if on_conflict not in ["raise", "nest"]:
        raise ValueError(f"Invalid on_conflict value: {on_conflict}")

    current_contexts = (_contexts.current or {}).copy()
    unique_context_names = set()
    for context in contexts:
        if not (cls := getattr(context, "__class__", None)):
            raise ValueError("Only class instances can be declared as contexts")
        name = get_cls_name_or_raise(cls)
        if name in unique_context_names:
            raise ValueError(f"Provided multiple concurrent contexts for {name}")
        unique_context_names.add(name)
        if name in current_contexts:
            if on_conflict == "raise":
                raise ValueError(f"Context {name} already provided")
        current_contexts[name] = context

    current_hooks = (_hooks.current or {}).copy()
    for hook in hooks:
        if not isinstance(hook, HookFn):
            raise ValueError("Hook was not decorated with @hook")
        name = get_fn_name_or_raise(hook.target)
        if name in current_hooks:
            current_hooks[name].append(hook)
        else:
            current_hooks[name] = [hook]

    current_mocks = (_mocks.current or {}).copy()
    unique_mock_names = set()
    for mock in mocks:
        if not isinstance(mock, MockFn):
            raise ValueError("Mock was not decorated with @mock")
        name = get_fn_name_or_raise(mock.target)
        if name in unique_mock_names:
            raise ValueError(f"Provided multiple concurrent mocks for {name}")
        unique_mock_names.add(name)
        if name in current_mocks:
            current_mocks[name] = mock
        else:
            current_mocks[name] = mock

    with _contexts.push(current_contexts):
        with _hooks.push(current_hooks):
            with _mocks.push(current_mocks):
                yield


@overload
def task(fn: F) -> F: ...


@overload
def task() -> (
    Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]
): ...


def task(
    fn: Callable[P, Coroutine[Any, Any, R]] | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    def decorator(
        fn: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            parent_observation: Observation | None = None
            try:
                parent_observation = use(Observation)
            except Exception:
                pass

            observation = Observation(
                id=uuid4(),
                parent=parent_observation,
                children=[],
            )
            if parent_observation:
                parent_observation.children.append(observation)

            with provide(observation, on_conflict="nest"):
                try:
                    hooks = _hooks.use(fn)
                except Exception:
                    hooks = []

                generators: list[Hook] = []
                injected_inputs: dict[str, Any] = {}
                for hook in hooks:
                    gen = hook(args, kwargs)
                    if isinstance(gen, Generator):
                        generators.append(gen)  # type: ignore[arg-type]
                        new_inputs = next(gen) or {}
                        injected_inputs.update(new_inputs)

                # rewrite task args/kwargs
                input_dict = format_input_dict(fn, args, kwargs)
                input_dict.update(injected_inputs)

                try:
                    mock = _mocks.use(fn)
                except Exception:
                    mock = None
                try:
                    if mock is not None:
                        result = await mock(**input_dict)
                    else:
                        raise MockMiss
                except MockMiss:  # make sure not to catch other errors
                    result = await fn(**input_dict)

                # send result to generator hooks
                for gen in generators:
                    try:
                        gen.send(result)
                    except StopIteration:
                        pass

                return result

        return wrapper

    if fn is not None:
        return decorator(fn)  # type: ignore[return-value]
    else:
        return decorator
