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
    ParamSpec,
    Type,
    TypeVar,
    cast,
    overload,
)

import nest_asyncio
import petname

from agentlens.cache import TaskCache
from agentlens.console import RunConsole
from agentlens.dataset import Dataset, Row
from agentlens.hooks import Hook
from agentlens.trace import Observation, Run
from agentlens.utils import now

_run_context: ContextVar[Run | None] = ContextVar("run_context", default=None)

F = TypeVar("F", bound=Callable[..., Any])
D = TypeVar("D", bound=Dataset)
P = ParamSpec("P")
R = TypeVar("R")


class Lens:
    _log = getLogger("agentlens")
    _dataset_dir: Path
    _runs_dir: Path

    def __init__(
        self,
        *,
        dataset_dir: Path | str,
        runs_dir: Path | str,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._runs_dir = Path(runs_dir)

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

            # conditionally cache
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
                self._handle_exception(observation, e)
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
                self._handle_exception(observation, e)
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
        main: Callable[[Row], Awaitable[R]] | Callable[[Row], R],
        dataset: Dataset,
        hooks: list[Callable[[Row], Hook]] | None = None,
    ) -> list[R]:
        nest_asyncio.apply()
        run_key = self._create_run_key()
        runs = []
        results: list[R] = []

        for idx, row in enumerate(dataset):
            runs.append(
                Run(
                    key=run_key,
                    dir=self._runs_dir / f"row_{idx}",
                    name=f"Row {idx}",
                    row=row,
                    hooks=self._initialize_hooks(row, hooks),
                )
            )

        async def execute():
            nonlocal results
            with TaskCache.enable(self._dataset_dir / "cache"):
                if asyncio.iscoroutinefunction(main):
                    results = await self._run_async(main, runs)
                else:
                    typed_main = cast(Callable[[Row], R], main)
                    results = self._run_sync(typed_main, runs)

        console = RunConsole(runs, execute)
        console.run()
        return results

    def _run_sync(
        self,
        main: Callable[[Row], R],
        runs: list[Run],
    ) -> list[R]:
        # maybe unnest if only one row
        results = []
        for run in runs:
            _run_context.set(run)
            result = main(run.row)
            results.append(result)
        return results

    def _run_async(
        self,
        main: Callable[[Row], Awaitable[R]],
        runs: list[Run],
    ) -> list[R]:
        # maybe unnest if only one row
        async def _run_one(run: Run) -> R:
            _run_context.set(run)
            return await main(run.row)

        async def _run_all() -> list[R]:
            tasks = [_run_one(run) for run in runs]
            return await asyncio.gather(*tasks)

        return asyncio.run(_run_all())

    def _initialize_hooks(
        self,
        row: Row,
        hook_factories: list[Callable[[Row], Hook]] | None,
    ) -> dict[str, list[Hook]]:
        hooks: dict[str, list[Hook]] = {}
        for hook_factory in hook_factories or []:
            hook = hook_factory(row)
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

    def _handle_exception(
        self,
        trace: Observation | None,
        e: Exception,
    ):
        if trace:
            trace.error = str(e)
            self._cleanup_observation(trace)
        raise e

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

    def hook(self, target_func: Callable, **kwargs) -> Callable[[Callable], Callable[[Row], Hook]]:
        def decorator(cb: Callable) -> Callable[[Row], Hook]:
            @wraps(cb)
            def wrapper(row: Row) -> Hook:
                return Hook(cb, target_func, row, **kwargs)

            return wrapper

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
