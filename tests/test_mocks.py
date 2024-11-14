import pytest

from agentlens.lens import Lens
from tests.conftest import mock_simple, mock_with_miss


@pytest.mark.asyncio
async def test_mock_simple(ls: Lens):
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
async def test_mock_miss(ls: Lens):
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
async def test_mock_context(ls: Lens):
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
async def test_nested_mock_contexts(ls: Lens):
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


@pytest.mark.asyncio
async def test_mock_with_hook(ls: Lens):
    """Test that hooks can control mocking behavior"""

    @ls.task(mock=mock_simple)
    async def target(x: int) -> int:
        return x * 2

    @ls.hook(target)
    def hook_with_mock(x: int):
        with ls.mock():
            result = yield {}
            assert result == x * 3  # mock result
        return None

    @ls.hook(target)
    def hook_with_no_mock(x: int):
        with ls.no_mock():
            result = yield {}
            assert result == x * 2  # original result
        return None

    # Test hook with mock
    with ls.provide(hooks=[hook_with_mock]):
        with ls.no_mock():
            result = await target(5)
            assert result == 15  # mock result

    # Test hook with no_mock
    with ls.provide(hooks=[hook_with_no_mock]):
        with ls.mock():
            result = await target(5)
            assert result == 10  # original result


@pytest.mark.asyncio
async def test_mock_precedence(ls: Lens):
    """Test that inner mock contexts take precedence"""

    @ls.task(mock=mock_simple)
    async def target(x: int) -> int:
        return x * 2

    @ls.hook(target)
    def hook_toggle_mock(x: int):
        yield {}  # Should use outer context

        with ls.mock():
            result2 = yield {}  # Should be mocked
            assert result2 == x * 3

        with ls.no_mock():
            result3 = yield {}  # Should be original
            assert result3 == x * 2

        return None

    # Test with outer mock context
    with ls.mock():
        with ls.provide(hooks=[hook_toggle_mock]):
            result = await target(5)
            assert result == 15  # Should use mock

    # Test with outer no_mock context
    with ls.no_mock():
        with ls.provide(hooks=[hook_toggle_mock]):
            result = await target(5)
            assert result == 10  # Should use original
