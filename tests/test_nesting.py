from dataclasses import dataclass, field

import pytest

from agentlens.client import provide, task, use


@dataclass
class Trace:
    __name__ = "Trace"
    children: list[str] = field(default_factory=list)


@task
async def child_task_one():
    parent_trace = use(Trace)
    parent_trace.children.append("child_one")
    return parent_trace


@task
async def child_task_two():
    parent_trace = use(Trace)
    parent_trace.children.append("child_two")
    return parent_trace


@pytest.mark.asyncio
async def test_conflict_raise():
    """
    Provide the same context type twice with on_conflict='raise' -> ValueError
    """
    t1 = Trace()
    t2 = Trace()
    with provide(t1):
        with pytest.raises(ValueError, match="Context Trace already provided"):
            with provide(t2, on_conflict="raise"):
                pass


@pytest.mark.asyncio
async def test_conflict_nest():
    """
    Provide the same context type twice with on_conflict='nest' -> overshadow, revert after.
    """
    parent_trace = Trace()
    child_trace = Trace()

    @task
    async def read_trace():
        return use(Trace)

    with provide(parent_trace):
        t = await read_trace()
        assert t is parent_trace

        with provide(child_trace, on_conflict="nest"):
            t2 = await read_trace()
            assert t2 is child_trace

        t3 = await read_trace()
        assert t3 is parent_trace


@pytest.mark.asyncio
async def test_mutate_parent_trace():
    """
    Child block overshadowing the parent's Trace can still hold a reference to
    mutate the parent if it grabs it first.
    """
    parent_trace = Trace()

    @task
    async def child():
        p = use(Trace)  # parent's trace
        new_trace = Trace()
        with provide(new_trace, on_conflict="nest"):
            p.children.append("mutated parent")
            return use(Trace)  # new_trace
        # outside block => parent's trace is again top

    with provide(parent_trace):
        returned_child_trace = await child()
        assert returned_child_trace is not parent_trace
        # but parent's trace was mutated
        assert parent_trace.children == ["mutated parent"]


@pytest.mark.asyncio
async def test_reading_from_parent():
    parent_trace = Trace()

    @task
    async def child_task():
        p = use(Trace)
        child_trace = Trace()
        with provide(child_trace, on_conflict="nest"):
            # overshadow => use(Trace) => child_trace
            p.children.append("child_was_here")
        return p

    with provide(parent_trace):
        t = await child_task()
        assert t is parent_trace
        assert "child_was_here" in parent_trace.children
