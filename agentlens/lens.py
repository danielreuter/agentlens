import asyncio
import inspect
from contextvars import ContextVar
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    ParamSpec,
    Sequence,
    Type,
    TypeVar,
    cast,
    overload,
)

import nest_asyncio
import petname

from agentlens.cache import TaskCache
from agentlens.console import RunConsole
from agentlens.dataset import Dataset, Example
from agentlens.hooks import Hook
from agentlens.trace import Observation, Run
from agentlens.utils import now

_run_context: ContextVar[Run | None] = ContextVar("run_context", default=None)

F = TypeVar("F", bound=Callable[..., Any])
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound=Example)


class Lens:
    _log = getLogger("agentlens")
    _dataset_dir: Path
    _runs_dir: Path

    def __init__(
        self,
        *,
        dataset_dir: Path | str,
        runs_dir: Path | str,
        editor: str = "cursor",
    ):
        self._dataset_dir = Path(dataset_dir)
        self._runs_dir = Path(runs_dir)
        self._editor = editor

    @overload
    def task(
        self,
        name: str | None = None,
        cache: bool = False,
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

    @overload  # type: ignore[misc]
    def task(
        self,
        name: str | None = None,
        cache: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def task(
        self,
        name: str | None = None,
        cache: bool = False,
    ) -> (
        Callable[[Callable[P, R]], Callable[P, R]]
        | Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]
    ):
        def decorator(func: Callable[P, R]) -> Callable[P, R] | Callable[P, Awaitable[R]]:
            task_name = name or func.__name__

            if cache:
                func = TaskCache.cached(func)

            if asyncio.iscoroutinefunction(func):
                typed_func = cast(Callable[P, Awaitable[R]], func)
                return self._async_task(typed_func, task_name)
            else:
                return self._sync_task(func, task_name)

        return decorator  # type: ignore[return-value]

    def _async_task(
        self,
        func: Callable[P, Awaitable[R]],
        name: str,
    ) -> Callable[P, Awaitable[R]]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            observation = self._make_observation(
                name=name,
                is_method=self._is_method(func),
                func_args=args,
                func_kwargs=kwargs,
            )
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if observation is not None:
                    observation.error = str(e)
                raise
            finally:
                self._cleanup_observation(observation)

        return async_wrapper

    def _sync_task(
        self,
        func: Callable[P, R],
        name: str,
    ) -> Callable[P, R]:
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            observation = self._make_observation(
                name=name,
                is_method=self._is_method(func),
                func_args=args,
                func_kwargs=kwargs,
            )
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if observation is not None:
                    observation.error = str(e)
                raise
            finally:
                self._cleanup_observation(observation)

        return sync_wrapper

    @staticmethod
    def _is_method(func: Callable) -> bool:
        params = inspect.signature(func).parameters
        return "self" in params or "cls" in params

    def run(
        self,
        *,
        main: Callable[[T], Awaitable[R]] | Callable[[T], R],
        dataset: Dataset,
        hooks: Sequence[Callable[[T], Hook]] | None = None,
    ) -> Sequence[R | None]:
        nest_asyncio.apply()
        run_key = self._create_run_key()
        run_dir = self._runs_dir / run_key
        runs = []
        results: Sequence[R | None] = []

        for idx, example in enumerate(dataset):
            runs.append(
                Run(
                    key=run_key,
                    dir=run_dir / f"example_{idx}",
                    name=f"Example {idx}",
                    example=example,
                    hooks=self._initialize_hooks(example, hooks),
                )
            )

        async def execute():
            nonlocal results
            with TaskCache.enable(self._dataset_dir / "cache"):
                if asyncio.iscoroutinefunction(main):
                    results = await self._run_async(main, runs)
                else:
                    typed_main = cast(Callable[[T], R], main)
                    results = self._run_sync(typed_main, runs)

        console = RunConsole(runs, execute, editor=self._editor)
        console.run()
        return results

    def _run_sync(
        self,
        main: Callable[[T], R],
        runs: list[Run[T]],
    ) -> list[R | None]:
        # maybe unnest if only one example
        results = []
        for run in runs:
            _run_context.set(run)
            try:
                result = main(run.example)
            except Exception as e:
                run.observation.error = str(e)
                result = None
            results.append(result)
        return results

    def _run_async(
        self,
        main: Callable[[T], Awaitable[R]],
        runs: list[Run[T]],
    ) -> list[R | None]:
        # maybe unnest if only one example
        async def _run_one(run: Run) -> R | None:
            _run_context.set(run)
            try:
                return await main(run.example)
            except Exception as e:
                run.observation.error = str(e)
                return None

        async def _run_all() -> list[R | None]:
            tasks = [_run_one(run) for run in runs]
            return await asyncio.gather(*tasks)

        return asyncio.run(_run_all())

    def _initialize_hooks(
        self,
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

    def _create_run_key(self) -> str:
        timestamp = now().strftime("%Y%m%d_%H%M%S")
        id = petname.generate(words=3, separator="_")
        return f"{timestamp}_{id}"

    def _make_observation(
        self,
        *,
        name: str,
        is_method: bool = False,
        func_args: tuple = (),
        func_kwargs: dict = {},
        cache: bool = False,
    ) -> Observation | None:
        # todo - determine cache hit, log inputs/outputs
        run = _run_context.get()
        if run is None:  # not in evaluation mode
            return None
        stack = run.observation_stack.copy()
        if not stack:
            raise ValueError("Observation stack unexpectedly empty")
        parent = stack[-1]
        observation = Observation(name=name)
        parent.add_child(observation)
        run.observation_stack = stack + [observation]
        return observation

    def _cleanup_observation(
        self,
        observation: Observation | None,
    ) -> None:
        if observation is None:
            return
        observation.end()
        run = _run_context.get()
        if run is None:
            raise ValueError("Observation stack unexpectedly empty")
        stack = run.observation_stack.copy()
        stack.pop()
        run.observation_stack = stack
        if not stack:
            _run_context.set(None)

    def hook(
        self,
        target_func: Callable,
        type: Literal["pre", "post"] = "post",
        **kwargs,
    ) -> Callable[[Callable], Callable[[T], Hook]]:
        def decorator(cb: Callable) -> Callable[[T], Hook]:
            @wraps(cb)
            def wrapper(example: T) -> Hook:
                return Hook(cb, target_func, example, type=type, **kwargs)

            return wrapper  # type: ignore[return-value]

        return decorator

    def score(self):
        pass

    def dataset(self, name: str) -> Callable[[Type[D]], Type[D]]:
        def decorator(cls: Type[D]) -> Type[D]:
            cls.name = name
            cls.dataset_dir = self._dataset_dir
            return cls

        return decorator

    # todo - nest this in observation
    def write_text(self, file_name: str, text: str):
        (self._runs_dir / file_name).write_text(text)

    def write_json(self, file_name: str, data: dict):
        pass

    def log(self, message: str) -> None:
        run = _run_context.get()
        if run is None:  # not in evaluation mode
            self._log.info(message)  # fallback to regular logging
            return

        stack = run.observation_stack
        if not stack:
            raise ValueError("Observation stack unexpectedly empty")
        current_observation = stack[-1]
        current_observation.add_log(message)

    def write(self, name: str, content: str) -> None:
        run = _run_context.get()
        if run is None:  # not in evaluation mode
            self._log.warning("Attempting to write file outside of run context")
            return

        stack = run.observation_stack
        if not stack:
            raise ValueError("Observation stack unexpectedly empty")

        current_observation = stack[-1]
        current_observation.add_file(name, content)

        filepath = run.dir / name
        filepath.write_text(content)

    def __truediv__(self, other: str | Path) -> Path:
        """Allow using / operator to create paths relative to run_dir."""
        run = _run_context.get()
        if run is None:
            raise ValueError("Cannot use path operations outside of run context")
        return run.dir / str(other)
