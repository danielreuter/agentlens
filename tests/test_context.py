import pytest

from agentlens.client import observe, provide, use
from tests.conftest import Counter, Messages


@observe
async def increment_counter():
    c = use(Counter)
    c.value += 1
    return c.value


@observe
async def read_counter():
    c = use(Counter)
    return c.value


@pytest.mark.asyncio
async def test_single_context():
    """
    Provide a single context and verify tasks can read/update it.
    """
    counter = Counter(value=0)
    with provide(counter):
        assert await increment_counter() == 1
        assert await increment_counter() == 2

    # Outside the block, context is gone
    # We expect "Context Counter not found" from your code
    with pytest.raises(ValueError, match="Context Counter not found"):
        await read_counter()


@pytest.mark.asyncio
async def test_multiple_contexts():
    """
    Provide two contexts. Each task uses whichever context it needs.
    """
    counter = Counter(value=10)
    messages = Messages()

    @observe
    async def add_message(msg: str) -> None:
        m = use(Messages)
        m.items.append(msg)

    with provide(counter, messages):
        assert await increment_counter() == 11
        await add_message("hello")
        assert messages.items == ["hello"]


@pytest.mark.asyncio
async def test_double_provision_raises():
    """
    Provide the same context type twice with on_conflict='raise' → expect ValueError.
    """
    c1 = Counter(value=0)
    c2 = Counter(value=5)
    with provide(c1):
        with pytest.raises(ValueError, match="Context Counter already provided"):
            with provide(c2, on_conflict="raise"):
                pass


@pytest.mark.asyncio
async def test_missing_context():
    """
    Trying to use a context not provided should raise ValueError.
    """
    # We expect "Context Counter not found"
    with pytest.raises(ValueError, match="Context Counter not found"):
        await read_counter()


@pytest.mark.asyncio
async def test_mutate_parent_through_reference():
    """
    Demonstrates that a child block can hold a reference to the parent's context
    and mutate it even if it provides a new context of the same type.
    """
    parent_counter = Counter(value=0)

    @observe
    async def child_task():
        # Grab the parent's counter before overshadowing
        parent_ref = use(Counter)
        new_counter = Counter(value=100)
        with provide(new_counter, on_conflict="nest"):
            # Inside here, use(Counter) == new_counter
            # but we can still mutate parent_ref
            parent_ref.value += 10
            assert use(Counter).value == 100
        # Once we exit, use(Counter) is parent_ref again
        return parent_ref.value

    with provide(parent_counter):
        val = await child_task()
        # The child added 10 to the parent's counter
        assert val == 10
        assert parent_counter.value == 10
