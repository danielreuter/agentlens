import asyncio
import time

import pytest

from agentlens.cache import TaskCache
from tests.conftest import MockInput, MockOutput


@pytest.mark.asyncio
async def test_basic_task_execution(ai_fixture):
    square = ai_fixture["square"]
    start_time = time.time()
    result = await square(5)
    duration = time.time() - start_time

    assert result == 25
    assert duration >= 0.1


@pytest.mark.asyncio
async def test_task_with_pydantic_models(ai_fixture):
    multiply = ai_fixture["multiply_with_logging"]
    input_data = MockInput(value=10, multiplier=1.5)

    start_time = time.time()
    result = await multiply(input_data)
    duration = time.time() - start_time

    assert result.result == 15.0
    assert len(result.steps) == 1
    assert "Multiplied 10 by 1.5" in result.steps[0]
    assert duration >= 0.1


@pytest.mark.asyncio
async def test_cached_task(ai, ai_fixture):
    with TaskCache.enable(ai._dataset_dir / "cache"):
        expensive_op = ai_fixture["expensive_operation"]

        # First call should take ~0.1 second
        start_time = time.time()
        result1 = await expensive_op(5)
        first_duration = time.time() - start_time

        # Second call should be fast (cached)
        start_time = time.time()
        result2 = await expensive_op(5)
        second_duration = time.time() - start_time

        assert result1 == result2 == 10
        assert first_duration >= 0.1
        assert second_duration < 0.09


@pytest.mark.asyncio
async def test_task_without_input_capture(ai_fixture):
    sensitive_op = ai_fixture["sensitive_operation"]

    start_time = time.time()
    result = await sensitive_op("password123")
    duration = time.time() - start_time

    assert result == "Processed: password123"
    assert duration >= 0.1


@pytest.mark.asyncio
async def test_task_without_output_capture(ai, ai_fixture):
    private_result = ai_fixture["private_result"]

    start_time = time.time()
    result = await private_result(42)
    duration = time.time() - start_time

    assert result == "Secret result: 42"
    assert duration >= 0.1


@pytest.mark.asyncio
async def test_composed_tasks(ai_fixture):
    complex_calc = ai_fixture["complex_calculation"]
    # For input 4:
    # 1. square(4) = 16
    # 2. expensive_operation(16) = 32
    # 3. multiply_with_logging(MockInput(value=32, multiplier=0.5)) = 16.0

    start_time = time.time()
    result = await complex_calc(4)
    duration = time.time() - start_time

    assert result == 16.0
    # Should take ~0.3 seconds (3 sequential tasks)
    assert duration >= 0.3


@pytest.mark.asyncio
async def test_task_in_async_context(ai_fixture):
    square = ai_fixture["square"]

    start_time = time.time()
    result = await square(6)
    duration = time.time() - start_time

    assert result == 36
    assert duration >= 0.1


@pytest.mark.asyncio
async def test_cache_invalidation_with_different_args(ai, ai_fixture):
    with TaskCache.enable(ai._dataset_dir / "cache"):
        expensive_op = ai_fixture["expensive_operation"]

        # First call - uncached
        start_time = time.time()
        result1 = await expensive_op(5)
        first_duration = time.time() - start_time

        # Different arg - should not use cache
        start_time = time.time()
        result2 = await expensive_op(6)
        second_duration = time.time() - start_time

        # Same as first arg - should use cache
        start_time = time.time()
        result3 = await expensive_op(5)
        third_duration = time.time() - start_time

        assert result1 == 10
        assert result2 == 12
        assert result3 == 10
        assert result1 == result3

        assert first_duration >= 0.1
        assert second_duration >= 0.1
        assert third_duration < 0.09


@pytest.mark.asyncio
async def test_cache_with_complex_args(ai, ai_fixture):
    with TaskCache.enable(ai._dataset_dir / "cache"):

        @ai.task(cache=True)
        async def complex_args_task(x: int, y: str, z: dict) -> str:
            await asyncio.sleep(0.1)
            return f"Processed: {x}-{y}-{z['key']}"

        # First call - uncached
        start_time = time.time()
        result1 = await complex_args_task(1, "test", {"key": "value"})
        first_duration = time.time() - start_time

        # Same args - should use cache
        start_time = time.time()
        result2 = await complex_args_task(1, "test", {"key": "value"})
        second_duration = time.time() - start_time

        # Different args - should not use cache
        start_time = time.time()
        result3 = await complex_args_task(1, "test", {"key": "different"})
        third_duration = time.time() - start_time

        assert result1 == result2
        assert result1 != result3

        assert first_duration >= 0.1
        assert second_duration < 0.09
        assert third_duration >= 0.1


@pytest.mark.asyncio
async def test_cache_with_pydantic_models(ai, ai_fixture):
    with TaskCache.enable(ai._dataset_dir / "cache"):
        multiply = ai_fixture["multiply_with_logging"]

        @ai.task(cache=True)
        async def cached_multiply(input: MockInput) -> MockOutput:
            return await multiply(input)

        input1 = MockInput(value=10, multiplier=1.5)
        input2 = MockInput(value=10, multiplier=1.5)
        input3 = MockInput(value=10, multiplier=2.0)

        # First call - uncached
        start_time = time.time()
        result1 = await cached_multiply(input1)
        first_duration = time.time() - start_time

        # Same input - should use cache
        start_time = time.time()
        result2 = await cached_multiply(input2)
        second_duration = time.time() - start_time

        # Different input - should not use cache
        start_time = time.time()
        result3 = await cached_multiply(input3)
        third_duration = time.time() - start_time

        assert result1.result == result2.result == 15.0
        assert result3.result == 20.0

        assert first_duration >= 0.1
        assert second_duration < 0.09
        assert third_duration >= 0.1


@pytest.mark.asyncio
async def test_cache_with_none_values(ai, ai_fixture):
    with TaskCache.enable(ai._dataset_dir / "cache"):

        @ai.task(cache=True)
        async def nullable_task(x: int | None) -> str:
            await asyncio.sleep(0.1)
            return f"Value: {x}"

        # First call with None - uncached
        start_time = time.time()
        result1 = await nullable_task(None)
        first_duration = time.time() - start_time

        # Second call with None - should use cache
        start_time = time.time()
        result2 = await nullable_task(None)
        second_duration = time.time() - start_time

        # Call with value - should not use cache
        start_time = time.time()
        result3 = await nullable_task(5)
        third_duration = time.time() - start_time

        assert result1 == result2 == "Value: None"
        assert result3 == "Value: 5"

        assert first_duration >= 0.1
        assert second_duration < 0.09
        assert third_duration >= 0.1


@pytest.mark.asyncio
async def test_cache_with_default_args(ai, ai_fixture):
    with TaskCache.enable(ai._dataset_dir / "cache"):

        @ai.task(cache=True)
        async def default_args_task(x: int, y: str = "default") -> str:
            await asyncio.sleep(0.1)
            return f"{x}-{y}"

        # First call with default - uncached
        start_time = time.time()
        result1 = await default_args_task(1)
        first_duration = time.time() - start_time

        # Second call with default - should use cache
        start_time = time.time()
        result2 = await default_args_task(1)
        second_duration = time.time() - start_time

        # Explicit default value - should use cache
        start_time = time.time()
        result3 = await default_args_task(1, "default")
        third_duration = time.time() - start_time

        # Different value - should not use cache
        start_time = time.time()
        result4 = await default_args_task(1, "different")
        fourth_duration = time.time() - start_time

        assert result1 == result2 == result3 == "1-default"
        assert result4 == "1-different"

        assert first_duration >= 0.1
        assert second_duration < 0.09
        assert third_duration < 0.09
        assert fourth_duration >= 0.1


@pytest.mark.asyncio
async def test_cache_performance(ai, ai_fixture):
    @ai.task(cache=True)
    async def slow_task(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * 2

    with TaskCache.enable(ai._dataset_dir / "cache"):
        # First call - should be slow
        start_time = time.time()
        result1 = await slow_task(5)
        first_duration = time.time() - start_time

        # Second call - should be fast (cached)
        start_time = time.time()
        result2 = await slow_task(5)
        second_duration = time.time() - start_time

        # Different input - should be slow
        start_time = time.time()
        result3 = await slow_task(6)
        third_duration = time.time() - start_time

        assert result1 == result2 == 10
        assert result3 == 12

        assert first_duration >= 0.1
        assert second_duration < 0.09
        assert third_duration >= 0.1


@pytest.mark.asyncio
async def test_task_with_user_profile(ai_fixture, user_profile):
    process_user = ai_fixture["process_user_profile"]

    start_time = time.time()
    result = await process_user(user_profile)
    duration = time.time() - start_time

    assert result["summary"] == "John Doe (30)"
    assert result["status"] == "active"
    assert result["performance_rating"] == 4.5
    assert duration >= 0.1


@pytest.mark.asyncio
async def test_cached_order_calculation(ai, ai_fixture, complex_order):
    with TaskCache.enable(ai._dataset_dir / "cache"):
        calc_total = ai_fixture["calculate_order_total"]

        # First calculation - uncached
        start_time = time.time()
        result1 = await calc_total(complex_order)
        first_duration = time.time() - start_time

        # Second calculation - should use cache
        start_time = time.time()
        result2 = await calc_total(complex_order)
        second_duration = time.time() - start_time

        assert result1 == result2 == 39.96
        assert first_duration >= 0.1
        assert second_duration < 0.09


@pytest.mark.asyncio
async def test_contact_info_validation(ai_fixture, contact_info):
    validate = ai_fixture["validate_contact_info"]

    start_time = time.time()
    result = await validate(contact_info)
    duration = time.time() - start_time

    assert "Email present" in result
    assert "Phone present" in result
    assert "Has 2 addresses" in result
    assert duration >= 0.1
