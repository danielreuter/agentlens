from typing import Generator

import pytest

import agentlens.evaluation as ev
from agentlens import task
from agentlens.client import Lens, provide


# Test tasks
@task
async def complex_task(a: int, b: str, c: bool = True) -> str:
    return f"{a}-{b}-{c}"


@task
async def simple_task(x: int) -> int:
    return x * 2


@task
async def task_with_kwargs(*, name: str, value: int) -> dict:
    return {"name": name, "value": value}


@task
async def task_with_defaults(x: int, y: str = "default") -> str:
    return f"{x}-{y}"


@task
async def task_raises_error() -> None:
    raise ValueError("Expected error")


@task
async def nested_task(x: int) -> int:
    return await simple_task(x)


# Test mocks
async def mock_simple(x: int) -> int:
    return x * 3


async def mock_partial_args(a: int) -> str:
    return f"mocked-{a}"


async def mock_with_miss(x: int) -> int:
    if x > 10:
        raise ev.MockMiss()
    return x * 4


# Test hooks
@ev.hook(simple_task)
def hook_simple(x: int) -> ev.HookGenerator[int]:
    result = yield {}
    assert isinstance(result, int)
    return None


@ev.hook(complex_task)
def hook_modify_args(a: int, b: str) -> ev.HookGenerator[str]:
    result = yield {"a": a * 2, "b": b.upper()}
    assert isinstance(result, str)
    return None


@ev.hook(complex_task)
def hook_partial_args(a: int) -> ev.HookGenerator[str]:
    result = yield {"a": a + 1}
    assert isinstance(result, str)
    return None


@ev.hook(task_with_kwargs)
def hook_kwargs_only(name: str) -> ev.HookGenerator[dict]:
    result = yield {"name": name.upper()}
    assert isinstance(result, dict)
    return None


@ev.hook(simple_task)
def hook_modify_result(x: int) -> ev.HookGenerator[int]:
    yield {}


@ev.hook(simple_task)
def hook_multiply(x: int) -> ev.HookGenerator[int]:
    yield {"x": x * 2}
    return None


@ev.hook(simple_task)
def hook_add(x: int) -> ev.HookGenerator[int]:
    yield {"x": x + 1}
    return None


# Tests for Hook validation
def test_hook_invalid_param():
    """Test that hooks with invalid parameters are rejected"""
    with pytest.raises(ValueError, match="Parameter 'invalid' does not exist"):

        @ev.hook(simple_task)
        def invalid_hook(invalid: int) -> Generator[dict, int, None]:
            yield {}


def test_hook_valid_subset():
    """Test that hooks can request a subset of parameters"""

    @ev.hook(complex_task)
    def valid_hook(a: int) -> Generator[dict, str, None]:
        yield {}


def test_hook_all_params():
    """Test that hooks can request all parameters"""

    @ev.hook(complex_task)
    def valid_hook(a: int, b: str, c: bool) -> Generator[dict, str, None]:
        yield {}


@task
@pytest.mark.asyncio
async def test_hook_no_modifications(lens: Lens):
    """Test that hooks work without modifying arguments"""
    result = await simple_task(5)
    assert result == 10

    with provide(hooks=[hook_simple]):
        result = await simple_task(5)
        assert result == 10


@task
@pytest.mark.asyncio
async def test_hook_modify_args(lens: Lens):
    """Test that hooks can modify arguments"""
    result = await complex_task(5, "test")
    assert result == "5-test-True"

    with provide(hooks=[hook_modify_args]):
        result = await complex_task(5, "test")
        assert result == "10-TEST-True"


@task
@pytest.mark.asyncio
async def test_hook_partial_args(lens: Lens):
    """Test that hooks can modify a subset of arguments"""
    result = await complex_task(5, "test")
    assert result == "5-test-True"

    with provide(hooks=[hook_partial_args]):
        result = await complex_task(5, "test")
        assert result == "6-test-True"


@task
@pytest.mark.asyncio
async def test_hook_kwargs(lens: Lens):
    """Test that hooks work with kwargs-only functions"""
    result = await task_with_kwargs(name="test", value=42)
    assert result == {"name": "test", "value": 42}

    with provide(hooks=[hook_kwargs_only]):
        result = await task_with_kwargs(name="test", value=42)
        assert result == {"name": "TEST", "value": 42}


# def test_mock_valid_subset():
#     """Test that mocks can use a subset of parameters"""

#     @task(mock=mock_partial_args)
#     async def target(a: int, b: str) -> str:
#         return f"{a}-{b}"


# # Tests for Mock beh@taskavior
# @pytest.mark.asyncio
# async def test_mock_simple():
#     """Test basic mock functionality"""

#     @task(mock=mock_simple)
#     async def target(x: int) -> int:
#         return x * 2

#     result = await target(5)
#     assert result == 10  # original

#     with ls.mock():
#         result = await target(5)
#         assert result == 15  # mock returns x * 3


@task
# @pytest.mark.asyncio
# async def test_mock_miss():
#     """Test MockMiss behavior"""

#     @task(mock=mock_with_miss)
#     async def target(x: int) -> int:
#         return x * 2

#     with ls.mock():
#         result = await target(5)
#         assert result == 20  # mock returns x * 4

#         result = await target(15)
#         assert result == 30  # MockMiss triggers original function

@task
# @pytest.mark.asyncio
# async def test_mock_context():
#     """Test mock context manager behavior"""

#     @task(mock=mock_simple)
#     async def target(x: int) -> int:
#         return x * 2

#     result1 = await target(5)
#     assert result1 == 10  # original

#     with ls.mock():
#         result2 = await target(5)
#         assert result2 == 15  # mocked

#     result3 = await target(5)
#     assert result3 == 10  # original again

@task
# @pytest.mark.asyncio
# async def test_nested_mock_contexts():
#     """Test nested mock context managers"""

#     @task(mock=mock_simple)
#     async def target(x: int) -> int:
#         return x * 2

#     with ls.mock():
#         result1 = await target(5)
#         assert result1 == 15  # mocked

#         with ls.no_mock():
#             result2 = await target(5)
#             assert result2 == 10  # original

#         result3 = await target(5)
#         assert result3 == 15  # mocked again


# New test cases@task
@pytest.mark.asyncio
async def test_hook_with_default_args(lens: Lens):
    """Test hooks work with functions that have default arguments"""
    result = await task_with_defaults(5)
    assert result == "5-default"

    @ev.hook(task_with_defaults)
    def hook_override_default(x: int) -> ev.HookGenerator[str]:
        yield {"y": "override"}
        return None

    with provide(hooks=[hook_override_default]):
        result = await task_with_defaults(5)
        assert result == "5-override"
