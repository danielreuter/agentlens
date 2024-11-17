from contextlib import contextmanager
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Iterator,
    ParamSpec,
    TypeVar,
    overload,
)

import tqdm
from tqdm.asyncio import tqdm_asyncio

from agentlens.constants import RUNS_DIR
from agentlens.context import Run
from agentlens.hooks import GeneratorHook, Hook, Mock, MockMiss, convert_to_kwargs

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R", covariant=True)
F = TypeVar("F", bound=Callable[..., Any])


def iter(self, iterable: Iterable[T], desc: str | None = None) -> Iterator[T]:
    return tqdm.tqdm(iterable, desc=desc)


async def gather(self, *coros: Awaitable[T], desc: str | None = None) -> list[T]:
    return await tqdm_asyncio.gather(*coros, desc=desc)


@contextmanager
def provide(
    *contexts: Any,
    hooks: list[Hook] = [],
) -> Generator[None, None, None]:
    with _provide_run() as run:
        with _provide_contexts(run, *contexts):
            with _provide_hooks(run, hooks):
                yield


@contextmanager
def _provide_run() -> Generator[Run, None, None]:
    """Manages the run context, creating a new one if needed."""
    run = Run.current()
    if run is None:
        run = Run.start(RUNS_DIR)
        yield run
        Run.end()
    else:
        yield run


@contextmanager
def _provide_contexts(run: Run, *contexts: Any) -> Generator[None, None, None]:
    context_keys = [type(context).__name__ for context in contexts]

    # save previous values
    current_contexts = run.store.contexts.get().copy()
    prev_values = {key: current_contexts.get(key) for key in context_keys}

    # set new values
    for context, key in zip(contexts, context_keys):
        if key in current_contexts:
            raise ValueError(f"Context {key} already provided")
        else:
            current_contexts[key] = context

    try:
        yield
    finally:
        # restore previous values
        for key, prev_value in prev_values.items():
            if prev_value is not None:
                current_contexts[key] = prev_value
            else:
                del current_contexts[key]


@contextmanager
def _provide_hooks(run: Run, hooks: list[Hook]) -> Generator[None, None, None]:
    """Provide hooks for the current run"""
    # Group hooks by their target function name
    hook_map: dict[str, list[Hook]] = {}
    for hook in hooks:
        target_name = hook.target.__name__
        if target_name not in hook_map:
            hook_map[target_name] = []
        hook_map[target_name].append(hook)

    # Store previous hooks
    current_hooks = run.store.hooks.get().copy()
    prev_hooks = current_hooks.copy()

    # Update run's hooks
    for target_name, target_hooks in hook_map.items():
        if target_name not in current_hooks:
            current_hooks[target_name] = []
        current_hooks[target_name].extend(target_hooks)

    try:
        yield
    finally:
        # Restore previous hooks
        run.store.hooks.set(prev_hooks)


@overload
def task(fn: F) -> F: ...


@overload
def task(
    fn: None = None,
    *,
    name: str | None = None,
    mock: Callable | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]: ...


def task(
    fn: Callable[P, Coroutine[Any, Any, R]] | None = None,
    *,
    name: str | None = None,
    mock: Callable | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    def decorator(
        fn: Callable[P, Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, R]]:
        if name:
            fn.__name__ = name

        if mock is not None:
            setattr(fn, "_mock_fn", Mock(mock, fn))

        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with _provide_run() as run:
                with run.create_observation(fn.__name__):
                    hooks = run.store.hooks.get().get(fn.__name__, [])

                    # run pre-hooks
                    generators: list[GeneratorHook] = []
                    injected_kwargs: dict[str, Any] = {}
                    for hook in hooks:
                        # should produce None or a generator
                        gen = hook(args, kwargs)
                        if isinstance(gen, Generator):
                            generators.append(gen)  # type: ignore[arg-type]
                            new_injected_kwargs = next(gen)
                            injected_kwargs.update(new_injected_kwargs)

                    # rewrite task args/kwargs
                    all_kwargs = convert_to_kwargs(fn, args, kwargs)
                    all_kwargs.update(injected_kwargs)

                    # execute task
                    mocks = run.store.mocks.get()
                    mock_fn = mocks.get(fn)
                    try:
                        if mock_fn is not None:
                            result = await mock_fn(**all_kwargs)
                        else:
                            raise MockMiss
                    except MockMiss:
                        result = await fn(**all_kwargs)

                    # send result to generator hooks
                    for gen in generators:
                        try:
                            gen.send(result)
                        except StopIteration:
                            pass

                    return result

        return wrapper

    return decorator
