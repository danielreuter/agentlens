from typing import (
    Awaitable,
    Iterable,
    Iterator,
    TypeVar,
)

from tqdm.asyncio import tqdm_asyncio

from agentlens.context import Run

T = TypeVar("T")


def iterate(iterable: Iterable[T], desc: str | None = None) -> Iterator[T]:
    for i, item in enumerate(iterable):
        Run.set_iteration(i)
        yield item
        Run.set_iteration(None)


async def gather(*coros: Awaitable[T], desc: str | None = None) -> list[T]:
    async def eval_coro(i: int, coro: Awaitable[T]) -> T:
        Run.set_iteration(i)
        try:
            return await coro
        finally:
            Run.set_iteration(None)

    tasks = [eval_coro(i, coro) for i, coro in enumerate(coros)]
    return await tqdm_asyncio.gather(*tasks, desc=desc)
