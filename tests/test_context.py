import pytest

from agentlens.client import Lens, provide, task
from tests.test_tasks import (
    Counter,
    Messages,
    add_message,
    get_counter_value,
    increment_counter,
    log_increment,
)


@pytest.mark.asyncio
@task
async def test_single_context(lens: Lens):
    """Test providing a single context"""
    counter = Counter(0)

    with provide(counter):
        result = await increment_counter()
        assert result == 1
        assert counter.value == 1


@pytest.mark.asyncio
@task
async def test_multiple_contexts(lens: Lens):
    """Test providing multiple contexts simultaneously"""
    counter = Counter(0)
    messages = Messages()

    with provide(counter, messages):
        await increment_counter()
        await add_message("test")

        assert counter.value == 1
        assert messages.items == ["test"]


@pytest.mark.asyncio
@task
async def test_double_provision_fails(lens: Lens):
    counter1 = Counter(0)
    counter2 = Counter(10)

    with provide(counter1):
        assert await get_counter_value() == 0

        with pytest.raises(ValueError, match="Context Counter already provided"):
            with provide(counter2):
                pass


@pytest.mark.asyncio
@task
async def test_context_with_hooks(lens: Lens):
    """Test contexts working with hooks"""
    counter = Counter(0)
    messages = Messages()

    with provide(counter, messages, hooks=[log_increment]):
        await increment_counter()

        assert counter.value == 1
        assert messages.items == ["Counter incremented to 1"]


@pytest.mark.asyncio
@task
async def test_missing_context(lens: Lens):
    """Test error when required context is missing"""
    with pytest.raises(ValueError, match="No context value provided for type Counter"):
        await get_counter_value()
