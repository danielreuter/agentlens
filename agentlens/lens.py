from logging import getLogger
from pathlib import Path
from typing import (
    Type,
    TypeVar,
    cast,
)

from agentlens.constants import RUNS_DIR
from agentlens.context import Run, Task

T = TypeVar("T")


class Lens:
    _log = getLogger("agentlens")
    _runs_dir: Path

    def __init__(
        self,
        *,
        runs_dir: Path = RUNS_DIR,
    ):
        self._runs_dir = Path(runs_dir)

    def __bool__(self) -> bool:
        try:
            Run.current()
            return True
        except ValueError:
            return False

    def __truediv__(self, other: str | Path) -> Path:
        run = Run.current()
        return run.dir / str(other)

    @property
    def run(self) -> Run:
        return Run.current()

    @property
    def task(self) -> Task:
        return Run.current_task()

    def __getitem__(self, context_type: Type[T]) -> T:
        key = context_type.__name__
        run = Run.current()
        contexts = run.store.contexts.get()
        try:
            return cast(T, contexts[key])
        except KeyError:
            raise ValueError(
                f"No context value provided for type {context_type.__name__}. "
                "Use 'with ls.provide(value):' to provide one."
            )


lens = Lens()
