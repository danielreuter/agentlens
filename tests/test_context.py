import pytest

from agentlens.lens import Lens
from tests.conftest import (
    Counter,
    Messages,
    UnregisteredContext,
    add_message,
    get_counter_value,
    increment_counter,
    log_increment,
)


@pytest.mark.asyncio
async def test_single_context(ls: Lens):
    """Test providing a single context"""
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(ls._context_types)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    counter = Counter(0)

    with ls.provide(counter):
        result = await increment_counter()
        assert result == 1
        assert counter.value == 1


@pytest.mark.asyncio
async def test_multiple_contexts(ls: Lens):
    """Test providing multiple contexts simultaneously"""
    counter = Counter(0)
    messages = Messages()

    with ls.provide(counter, messages):
        await increment_counter()
        await add_message("test")

        assert counter.value == 1
        assert messages.items == ["test"]


@pytest.mark.asyncio
async def test_nested_contexts(ls: Lens):
    """Test nested context providers"""
    counter1 = Counter(0)
    counter2 = Counter(10)

    with ls.provide(counter1):
        assert await get_counter_value() == 0

        with ls.provide(counter2):
            assert await get_counter_value() == 10

        assert await get_counter_value() == 0


@pytest.mark.asyncio
async def test_context_with_hooks(ls: Lens):
    """Test contexts working with hooks"""
    counter = Counter(0)
    messages = Messages()

    with ls.provide(counter, messages, hooks=[log_increment]):
        await increment_counter()

        assert counter.value == 1
        assert messages.items == ["Counter incremented to 1"]


@pytest.mark.asyncio
async def test_missing_context(ls: Lens):
    """Test error when required context is missing"""
    with pytest.raises(ValueError, match="No context value provided for type Counter"):
        await get_counter_value()


@pytest.mark.asyncio
async def test_unregistered_context(ls: Lens):
    """Test error when using unregistered context type"""
    context = UnregisteredContext(0)
    with pytest.raises(ValueError, match="has not been registered as a context"):
        with ls.provide(context):
            pass
