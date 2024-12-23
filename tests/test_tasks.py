import pytest

from agentlens.client import task


@task
async def add_nums(a: int, b: int) -> int:
    return a + b


@task
async def kw_only_task(*, name: str, value: int) -> dict:
    return {"name": name, "value": value}


@task
async def task_with_defaults(x: int, y: str = "default") -> str:
    return f"{x}-{y}"


@task
async def always_raises():
    raise ValueError("Expected error")


@task
async def parent_task():
    return "parent"


@task
async def child_task():
    return "child"


async def test_task_with_positional_args():
    result = await add_nums(3, 4)
    assert result == 7


async def test_task_with_keyword_only_args():
    result = await kw_only_task(name="alice", value=42)
    assert result == {"name": "alice", "value": 42}


async def test_task_with_defaults():
    assert await task_with_defaults(5) == "5-default"
    assert await task_with_defaults(5, "xyz") == "5-xyz"


async def test_task_raises_exception():
    with pytest.raises(ValueError, match="Expected error"):
        await always_raises()


async def test_nested_tasks():
    """
    Simple example: parent calls child. The framework
    may or may not track Observations, but the calls should succeed.
    """
    p_result = await parent_task()
    c_result = await child_task()
    assert p_result == "parent"
    assert c_result == "child"
