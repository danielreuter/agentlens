from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Iterable,
    Iterator,
    ParamSpec,
    Type,
    TypeVar,
    cast,
    overload,
)

import tqdm
from tqdm.asyncio import tqdm_asyncio

from agentlens.hooks import GeneratorHook, Hook, Mock, MockMiss, convert_to_kwargs
from agentlens.trace import Run

_run_context: ContextVar[Run | None] = ContextVar("run_context", default=None)
_do_mock: ContextVar[bool] = ContextVar("do_mock", default=False)
_do_cache: ContextVar[bool] = ContextVar("do_cache", default=False)

F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R", covariant=True)
T = TypeVar("T")
C = TypeVar("C")


class Lens:
    _log = getLogger("agentlens")
    _dataset_dir: Path
    _runs_dir: Path
    _context_types: dict[str, Type]
    _mock_registry: dict[Callable, Callable]

    def __init__(
        self,
        *,
        dataset_dir: Path | str,
        runs_dir: Path | str,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._runs_dir = Path(runs_dir)
        self._context_types = {}
        self._mock_registry = {}

    @overload
    def task(
        self, fn: Callable[P, Coroutine[Any, Any, R]]
    ) -> Callable[P, Coroutine[Any, Any, R]]: ...

    @overload
    def task(
        self,
        *,
        name: str | None = None,
        mock: Callable | None = None,
    ) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]: ...

    def task(
        self,
        fn: Callable[P, Coroutine[Any, Any, R]] | None = None,
        *,
        name: str | None = None,
        mock: Callable | None = None,
    ) -> (
        Callable[P, Coroutine[Any, Any, R]]
        | Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]
    ):
        def decorator(
            fn: Callable[P, Coroutine[Any, Any, R]],
        ) -> Callable[P, Coroutine[Any, Any, R]]:
            if name:
                fn.__name__ = name

            if mock is not None:
                setattr(fn, "_mock_fn", Mock(mock, fn))

            @wraps(fn)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with self._provide_run(fn.__name__) as run:
                    with run.create_observation(fn.__name__):
                        hooks = run.hooks.get(fn.__name__, [])

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
                        mock_fn = getattr(fn, "_mock_fn", None)
                        try:
                            if run.do_mock and mock_fn is not None:
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

    def hook(self, target_fn: Callable[..., Awaitable[Any]]) -> Callable[[Callable], Hook]:
        def decorator(hook_fn: Callable) -> Hook:
            if not hasattr(hook_fn, "__name__"):
                raise ValueError("Hook function must have a __name__ attribute")
            return Hook(hook_fn, target_fn)

        return decorator

    @property
    def root(self) -> Path:
        run = self._get_current_run()
        return run.dir

    def _get_current_run(self) -> Run:
        run = _run_context.get()
        if run is None:
            raise ValueError("Attempted to access current Run outside of a run context")
        return run

    def __truediv__(self, other: str | Path) -> Path:
        return self.root / str(other)

    def iter(self, iterable: Iterable[T], desc: str | None = None) -> Iterator[T]:
        return tqdm.tqdm(iterable, desc=desc)

    async def gather(self, *coros: Awaitable[T], desc: str | None = None) -> list[T]:
        return await tqdm_asyncio.gather(*coros, desc=desc)

    def context(self, cls: Type[C]) -> Type[C]:
        key = cls.__name__
        if key in self._context_types:
            existing_cls = self._context_types[key]
            if existing_cls != cls:
                raise ValueError(
                    f"Context name '{key}' is already registered to {existing_cls.__name__}. "
                    f"Cannot register it again for {cls.__name__}"
                )

        # Register the context type
        self._context_types[key] = cls
        return cls

    @contextmanager
    def provide(
        self,
        *contexts: Any,
        hooks: list[Hook] = [],
    ) -> Generator[None, None, None]:
        with self._provide_run() as run:
            with self._provide_contexts(run, *contexts):
                with self._provide_hooks(run, hooks):
                    yield

    @contextmanager
    def _provide_run(self, task_name: str | None = None) -> Generator[Run, None, None]:
        """Manages the run context, creating a new one if needed."""
        run = _run_context.get()
        created_run = False

        if run is None:
            if task_name is None:
                task_name = "default"
            run = Run(self._runs_dir, task_name)
            _run_context.set(run)
            created_run = True

        try:
            yield run
        finally:
            if created_run:
                _run_context.set(None)

    @contextmanager
    def _provide_contexts(self, run: Run, *contexts: Any) -> Generator[None, None, None]:
        context_keys = [type(context).__name__ for context in contexts]

        # ensure that all contexts have been registered
        for key in context_keys:
            if key not in self._context_types:
                raise ValueError(
                    f"Type {key} has not been registered as a context. "
                    "Use the `@ls.context` decorator to register it."
                )

        # save previous values
        prev_values = {key: run.context_values.get(key) for key in context_keys}

        # set new values
        for context, key in zip(contexts, context_keys):
            run.context_values[key] = context

        try:
            yield
        finally:
            # restore previous values
            for key, prev_value in prev_values.items():
                if prev_value is not None:
                    run.context_values[key] = prev_value
                else:
                    del run.context_values[key]

    @contextmanager
    def _provide_hooks(self, run: Run, hooks: list[Hook]) -> Generator[None, None, None]:
        """Provide hooks for the current run"""
        # Group hooks by their target function name
        hook_map: dict[str, list[Hook]] = {}
        for hook in hooks:
            target_name = hook.target.__name__
            if target_name not in hook_map:
                hook_map[target_name] = []
            hook_map[target_name].append(hook)

        # Store previous hooks
        prev_hooks = run.hooks.copy()

        # Update run's hooks
        for target_name, target_hooks in hook_map.items():
            if target_name not in run.hooks:
                run.hooks[target_name] = []
            run.hooks[target_name].extend(target_hooks)

        try:
            yield
        finally:
            # Restore previous hooks
            run.hooks = prev_hooks

    def __getitem__(self, context_type: Type[T]) -> T:
        """Get a context value by its type"""
        key = context_type.__name__
        run = self._get_current_run()
        try:
            return cast(T, run.context_values[key])
        except KeyError:
            raise ValueError(
                f"No context value provided for type {context_type.__name__}. "
                "Use 'with ls.provide(value):' to provide one."
            )

    def eval(self):  # type: ignore[no-untyped-def]
        """Just an alias for task for the extension to consume"""
        return self.task

    @contextmanager
    def mock(self):
        """Request that tasks in this context use their mock implementations"""
        with self._provide_run() as run:
            prev_mock = run.do_mock
            run.do_mock = True
            try:
                yield
            finally:
                run.do_mock = prev_mock

    @contextmanager
    def no_mock(self):
        """Request that tasks in this context use their real implementations"""
        with self._provide_run() as run:
            prev_mock = run.do_mock
            run.do_mock = False
            try:
                yield
            finally:
                run.do_mock = prev_mock

    @contextmanager
    def cache(self):
        """Request that tasks in this context use cached results if available"""
        token = _do_cache.set(True)
        try:
            yield
        finally:
            _do_cache.reset(token)

    @contextmanager
    def no_cache(self):
        """Request that tasks in this context ignore cached results"""
        token = _do_cache.set(False)
        try:
            yield
        finally:
            _do_cache.reset(token)
