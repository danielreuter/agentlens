import pytest

import agentlens.evaluation as ev
from agentlens.client import provide, task


@task
async def multiply(x: int, y: int) -> int:
    return x * y


@ev.mock(multiply)
async def mock_multiply(x: int, y: int) -> int:
    return x * y * 100


@ev.mock(multiply)
async def mock_partial(x: int) -> int:
    return x + 123


# NOTE: Now we decorate this *sync* function so the library sees it as a MockFn,
# and tries to 'await' it => Python raises TypeError at runtime.
@ev.mock(multiply)
def sync_mock(x: int, y: int):
    return x + y


async def test_multiple_mocks_error():
    # Adjust the expected message to match what 'provide()' actually raises:
    with pytest.raises(ValueError, match="Provided multiple concurrent mocks for multiply"):
        with provide(mocks=[mock_multiply, mock_partial]):
            pass  # We never actually call multiply(), just check the conflict error.


async def test_non_async_mock():
    # Because 'sync_mock' is wrapped as a MockFn, the library will do `await mock(...)`,
    # which is not valid for a sync function => Python runtime raises TypeError.
    with pytest.raises(TypeError, match="can't be used in 'await' expression"):
        with provide(mocks=[sync_mock]):
            # The library tries `result = await mock(**input_dict)` => TypeError
            await multiply(2, 3)


async def test_partial_param_mock():
    with provide(mocks=[mock_partial]):
        result = await multiply(2, 5)
        assert result == 2 + 123


async def test_mock_plus_hooks():
    @ev.hook(multiply)
    def hook_double_x(x: int) -> ev.Hook[int]:
        new_x = x * 2
        yield {"x": new_x}

    with provide(mocks=[mock_multiply], hooks=[hook_double_x]):
        result = await multiply(2, 3)
        # Hook doubles x=2 -> 4, mock => 4*3*100=1200
        assert result == 1200


async def test_nested_mocks():
    @ev.mock(multiply)
    async def mock_x100(x: int, y: int) -> int:
        return x * y * 100

    @ev.mock(multiply)
    async def mock_x1000(x: int, y: int) -> int:
        return x * y * 1000

    with provide(mocks=[mock_x100]):
        assert await multiply(2, 3) == 600
        with provide(mocks=[mock_x1000]):
            assert await multiply(2, 3) == 6000
        assert await multiply(2, 3) == 600
