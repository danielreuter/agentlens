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
async def task_raises_error() -> None:
    raise ValueError("Expected error")


# Test mocks
@ev.mock(simple_task)
async def mock_simple(x: int) -> int:
    return x * 3


@ev.mock(complex_task)
async def mock_partial_args(a: int) -> str:
    return f"mocked-{a}"


@ev.mock(simple_task)
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


# Tests for Mock validation
@task
@pytest.mark.asyncio
async def test_mock_invalid_param(lens: Lens):
    """Test that mocks with invalid parameters are rejected"""
    with pytest.raises(ValueError, match="Parameter 'invalid' does not exist"):

        @ev.mock(simple_task)
        async def invalid_mock(invalid: int) -> int:
            return 42


@task
@pytest.mark.asyncio
async def test_mock_must_be_async(lens: Lens):
    """Test that non-async mock functions are rejected"""
    with pytest.raises(TypeError, match="object int can't be used in 'await' expression"):

        @ev.mock(simple_task)
        def sync_mock(x: int) -> int:
            return x * 2

        with provide(mocks=[sync_mock]):
            await simple_task(5)


@task
@pytest.mark.asyncio
async def test_mock_return_type(lens: Lens):
    """Test that mocks with wrong return type are rejected"""
    with pytest.raises(TypeError, match="Mock callback must return same type as target"):

        @ev.mock(simple_task)
        async def wrong_return(x: int) -> str:
            return "wrong"


@task  # Tests for Mock behavior
@pytest.mark.asyncio
async def test_mock_simple(lens: Lens):
    """Test basic mock functionality"""
    result = await simple_task(5)
    assert result == 10  # original

    with provide(mocks=[mock_simple]):
        result = await simple_task(5)
        assert result == 15  # mock returns x * 3


@task
@pytest.mark.asyncio
async def test_mock_miss(lens: Lens):
    """Test MockMiss behavior"""
    with provide(mocks=[mock_with_miss]):
        result = await simple_task(5)
        assert result == 20  # mock returns x * 4

        result = await simple_task(15)
        assert result == 30  # MockMiss triggers original function


@task
@pytest.mark.asyncio
async def test_mock_partial_args(lens: Lens):
    """Test that mocks can use a subset of parameters"""
    with provide(mocks=[mock_partial_args]):
        result = await complex_task(5, "test")
        assert result == "mocked-5"


@task
@pytest.mark.asyncio
async def test_multiple_mocks_error(lens: Lens):
    """Test that providing multiple mocks for the same function raises an error"""
    with pytest.raises(ValueError, match="Multiple mocks provided for function simple_task"):
        with provide(mocks=[mock_simple, mock_with_miss]):
            pass


@task  # Tests for combined Mock and Hook behavior
@pytest.mark.asyncio
async def test_mocks_and_hooks(lens: Lens):
    """Test that mocks and hooks can work together"""
    with provide(mocks=[mock_simple], hooks=[hook_simple]):
        result = await simple_task(5)
        assert result == 15  # mock is applied, hook observes but doesn't modify


@task
@pytest.mark.asyncio
async def test_mock_with_arg_modifying_hook(lens: Lens):
    """Test that hooks can modify arguments before they reach the mock"""
    with provide(mocks=[mock_partial_args], hooks=[hook_modify_args]):
        result = await complex_task(5, "test")
        # Hook doubles 'a' (5 -> 10) before mock sees it
        assert result == "mocked-10"


@task
@pytest.mark.asyncio
async def test_mock_miss_with_hooks(lens: Lens):
    """Test that hooks still work when a mock misses"""
    with provide(mocks=[mock_with_miss], hooks=[hook_simple]):
        # For x=15, mock misses but hook still runs
        result = await simple_task(15)
        assert result == 30  # original function * 2


@task
@pytest.mark.asyncio
async def test_nested_contexts(lens: Lens):
    """Test nested mock/hook contexts"""
    with provide(mocks=[mock_simple]):
        result = await simple_task(5)
        assert result == 15  # using first mock

        with provide(mocks=[mock_with_miss]):
            result = await simple_task(5)
            assert result == 20  # using second mock

        result = await simple_task(5)
        assert result == 15  # back to first mock


@task  # Additional tests from test_hooks.py
@pytest.mark.asyncio
async def test_mock_valid_subset(lens: Lens):
    """Test that mocks can use a subset of parameters"""

    @task
    async def target(a: int, b: str) -> str:
        return f"{a}-{b}"

    @ev.mock(target)
    async def mock_subset(a: int) -> str:
        return f"mocked-{a}"

    with provide(mocks=[mock_subset]):
        result = await target(5, "test")
        assert result == "mocked-5"


@task
@pytest.mark.asyncio
async def test_mock_simple_basic(lens: Lens):
    """Test basic mock functionality"""

    @task
    async def target(x: int) -> int:
        return x * 2

    @ev.mock(target)
    async def mock_fn(x: int) -> int:
        return x * 3

    result = await target(5)
    assert result == 10  # original

    with provide(mocks=[mock_fn]):
        result = await target(5)
        assert result == 15  # mock returns x * 3


@task
@pytest.mark.asyncio
async def test_mock_miss_behavior(lens: Lens):
    """Test MockMiss behavior"""

    @task
    async def target(x: int) -> int:
        return x * 2

    @ev.mock(target)
    async def mock_fn(x: int) -> int:
        if x > 10:
            raise ev.MockMiss()
        return x * 4

    with provide(mocks=[mock_fn]):
        result = await target(5)
        assert result == 20  # mock returns x * 4

        result = await target(15)
        assert result == 30  # MockMiss triggers original function


@task
@pytest.mark.asyncio
async def test_mock_context_behavior(lens: Lens):
    """Test mock context manager behavior"""

    @task
    async def target(x: int) -> int:
        return x * 2

    @ev.mock(target)
    async def mock_fn(x: int) -> int:
        return x * 3

    result1 = await target(5)
    assert result1 == 10  # original

    with provide(mocks=[mock_fn]):
        result2 = await target(5)
        assert result2 == 15  # mocked

    result3 = await target(5)
    assert result3 == 10  # original again


@task
@pytest.mark.asyncio
async def test_nested_mock_contexts(lens: Lens):
    """Test nested mock context managers"""

    @task
    async def target(x: int) -> int:
        return x * 2

    @ev.mock(target)
    async def mock_fn(x: int) -> int:
        return x * 3

    @ev.mock(target)
    async def other_mock(x: int) -> int:
        return x * 4

    with provide(mocks=[mock_fn]):
        result1 = await target(5)
        assert result1 == 15  # first mock

        # Note: This will now raise an error due to multiple mocks
        # Instead of the original behavior, we'll test a different scenario
        result2 = await target(5)
        assert result2 == 15  # same mock still active

        with provide(mocks=[other_mock]):
            result3 = await target(5)
            assert result3 == 20  # second mock

        result4 = await target(5)
        assert result4 == 15  # back to first mock
