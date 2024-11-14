from typing import Generator

import pytest

from agentlens.hooks import GeneratorHook, MockMiss
from agentlens.lens import Lens

# Initialize test Lens
ls = Lens(dataset_dir="tests/datasets", runs_dir="tests/runs")


# Test tasks
@ls.task()
async def complex_task(a: int, b: str, c: bool = True) -> str:
    return f"{a}-{b}-{c}"


@ls.task()
async def simple_task(x: int) -> int:
    return x * 2


@ls.task()
async def task_with_kwargs(*, name: str, value: int) -> dict:
    return {"name": name, "value": value}


@ls.task()
async def task_with_defaults(x: int, y: str = "default") -> str:
    return f"{x}-{y}"


@ls.task()
async def task_raises_error() -> None:
    raise ValueError("Expected error")


@ls.task()
async def nested_task(x: int) -> int:
    return await simple_task(x)


# Test mocks
async def mock_simple(x: int) -> int:
    return x * 3


async def mock_partial_args(a: int) -> str:
    return f"mocked-{a}"


async def mock_with_miss(x: int) -> int:
    if x > 10:
        raise MockMiss()
    return x * 4


# Test hooks
@ls.hook(simple_task)
def hook_simple(x: int) -> GeneratorHook[int]:
    result = yield {}
    assert isinstance(result, int)
    return None


@ls.hook(complex_task)
def hook_modify_args(a: int, b: str) -> GeneratorHook[str]:
    result = yield {"a": a * 2, "b": b.upper()}
    assert isinstance(result, str)
    return None


@ls.hook(complex_task)
def hook_partial_args(a: int) -> GeneratorHook[str]:
    result = yield {"a": a + 1}
    assert isinstance(result, str)
    return None


@ls.hook(task_with_kwargs)
def hook_kwargs_only(name: str) -> GeneratorHook[dict]:
    result = yield {"name": name.upper()}
    assert isinstance(result, dict)
    return None


@ls.hook(simple_task)
def hook_modify_result(x: int) -> GeneratorHook[int]:
    yield {}


@ls.hook(simple_task)
def hook_multiply(x: int) -> GeneratorHook[int]:
    yield {"x": x * 2}
    return None


@ls.hook(simple_task)
def hook_add(x: int) -> GeneratorHook[int]:
    yield {"x": x + 1}
    return None


# Tests for Hook validation
def test_hook_invalid_param():
    """Test that hooks with invalid parameters are rejected"""
    with pytest.raises(ValueError, match="Parameter 'invalid' does not exist"):

        @ls.hook(simple_task)
        def invalid_hook(invalid: int) -> Generator[dict, int, None]:
            yield {}


def test_hook_valid_subset():
    """Test that hooks can request a subset of parameters"""

    @ls.hook(complex_task)
    def valid_hook(a: int) -> Generator[dict, str, None]:
        yield {}


def test_hook_all_params():
    """Test that hooks can request all parameters"""

    @ls.hook(complex_task)
    def valid_hook(a: int, b: str, c: bool) -> Generator[dict, str, None]:
        yield {}


# Tests for Hook argument handling
@pytest.mark.asyncio
async def test_hook_no_modifications(ls: Lens):
    """Test that hooks work without modifying arguments"""
    result = await simple_task(5)
    assert result == 10

    with ls.provide(hooks=[hook_simple]):
        result = await simple_task(5)
        assert result == 10


@pytest.mark.asyncio
async def test_hook_modify_args(ls: Lens):
    """Test that hooks can modify arguments"""
    result = await complex_task(5, "test")
    assert result == "5-test-True"

    with ls.provide(hooks=[hook_modify_args]):
        result = await complex_task(5, "test")
        assert result == "10-TEST-True"


@pytest.mark.asyncio
async def test_hook_partial_args(ls: Lens):
    """Test that hooks can modify a subset of arguments"""
    result = await complex_task(5, "test")
    assert result == "5-test-True"

    with ls.provide(hooks=[hook_partial_args]):
        result = await complex_task(5, "test")
        assert result == "6-test-True"


@pytest.mark.asyncio
async def test_hook_kwargs(ls: Lens):
    """Test that hooks work with kwargs-only functions"""
    result = await task_with_kwargs(name="test", value=42)
    assert result == {"name": "test", "value": 42}

    with ls.provide(hooks=[hook_kwargs_only]):
        result = await task_with_kwargs(name="test", value=42)
        assert result == {"name": "TEST", "value": 42}


# Tests for Mock validation
def test_mock_invalid_param():
    """Test that mocks with invalid parameters are rejected"""
    with pytest.raises(ValueError, match="Parameter 'invalid' does not exist"):

        @ls.task(mock=lambda invalid: None)
        async def target(x: int) -> int:
            return x


def test_mock_valid_subset():
    """Test that mocks can use a subset of parameters"""

    @ls.task(mock=mock_partial_args)
    async def target(a: int, b: str) -> str:
        return f"{a}-{b}"


# Tests for Mock behavior
@pytest.mark.asyncio
async def test_mock_simple():
    """Test basic mock functionality"""

    @ls.task(mock=mock_simple)
    async def target(x: int) -> int:
        return x * 2

    result = await target(5)
    assert result == 10  # original

    with ls.mock():
        result = await target(5)
        assert result == 15  # mock returns x * 3


@pytest.mark.asyncio
async def test_mock_miss():
    """Test MockMiss behavior"""

    @ls.task(mock=mock_with_miss)
    async def target(x: int) -> int:
        return x * 2

    with ls.mock():
        result = await target(5)
        assert result == 20  # mock returns x * 4

        result = await target(15)
        assert result == 30  # MockMiss triggers original function


@pytest.mark.asyncio
async def test_mock_context():
    """Test mock context manager behavior"""

    @ls.task(mock=mock_simple)
    async def target(x: int) -> int:
        return x * 2

    result1 = await target(5)
    assert result1 == 10  # original

    with ls.mock():
        result2 = await target(5)
        assert result2 == 15  # mocked

    result3 = await target(5)
    assert result3 == 10  # original again


@pytest.mark.asyncio
async def test_nested_mock_contexts():
    """Test nested mock context managers"""

    @ls.task(mock=mock_simple)
    async def target(x: int) -> int:
        return x * 2

    with ls.mock():
        result1 = await target(5)
        assert result1 == 15  # mocked

        with ls.no_mock():
            result2 = await target(5)
            assert result2 == 10  # original

        result3 = await target(5)
        assert result3 == 15  # mocked again


# New test cases
@pytest.mark.asyncio
async def test_hook_with_default_args(ls: Lens):
    """Test hooks work with functions that have default arguments"""
    result = await task_with_defaults(5)
    assert result == "5-default"

    @ls.hook(task_with_defaults)
    def hook_override_default(x: int) -> GeneratorHook[str]:
        yield {"y": "override"}
        return None

    with ls.provide(hooks=[hook_override_default]):
        result = await task_with_defaults(5)
        assert result == "5-override"
