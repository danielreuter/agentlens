import pytest

from agentlens import AI
from agentlens.dataset import Dataset, Row
from tests.conftest import MockInput


class TestRow(Row):
    value: int


class CaptureRow(Row):
    x: int
    expected_output: int


@pytest.mark.asyncio
async def test_basic_hook(ai, ai_fixture):
    square = ai_fixture["square"]
    results = []

    @ai.hook(square)
    def capture_result(row: TestRow, output, x):
        results.append((row.value, x, output))

    @ai.dataset("test")
    class TestDataset(Dataset[TestRow]): ...

    dataset = TestDataset()
    dataset.extend([TestRow(value=5)])

    async def main(row: TestRow):
        return await square(row.value)

    ai.run(
        main=main,
        hooks=[capture_result],
        dataset=dataset,
    )

    assert results == [(5, 5, 25)]


@pytest.mark.asyncio
async def test_multiple_hooks_same_task(ai, ai_fixture):
    square = ai_fixture["square"]
    inputs = []
    outputs = []

    @ai.hook(square)
    def capture_input(row: TestRow, output, x):
        inputs.append((row.value, x))

    @ai.hook(square)
    def capture_output(row: TestRow, output, x):
        outputs.append((row.value, output))

    @ai.dataset("test")
    class TestDataset(Dataset[TestRow]): ...

    dataset = TestDataset()
    dataset.extend([TestRow(value=5)])

    async def main(row: TestRow):
        return await square(row.value)

    ai.run(
        main=main,
        hooks=[capture_input, capture_output],
        dataset=dataset,
    )

    assert inputs == [(5, 5)]
    assert outputs == [(5, 25)]


@pytest.mark.asyncio
async def test_hook_with_pydantic_models(ai, ai_fixture):
    multiply = ai_fixture["multiply_with_logging"]
    results = []

    class MultiplyRow(Row):
        value: int
        multiplier: float

    @ai.hook(multiply)
    def capture_multiplication(row: MultiplyRow, output, input: MockInput):
        results.append(
            {
                "row_value": row.value,
                "row_multiplier": row.multiplier,
                "input_value": input.value,
                "multiplier": input.multiplier,
                "result": output.result,
            }
        )

    @ai.dataset("test")
    class TestDataset(Dataset[MultiplyRow]): ...

    dataset = TestDataset()
    dataset.extend([MultiplyRow(value=10, multiplier=1.5)])

    async def main(row: MultiplyRow):
        input_data = MockInput(value=row.value, multiplier=row.multiplier)
        return await multiply(input_data)

    ai.run(
        main=main,
        hooks=[capture_multiplication],
        dataset=dataset,
    )

    assert len(results) == 1
    assert results[0] == {
        "row_value": 10,
        "row_multiplier": 1.5,
        "input_value": 10,
        "multiplier": 1.5,
        "result": 15.0,
    }


# @pytest.mark.asyncio
# async def test_hooks_with_composed_tasks(ai, ai_fixture):
#     complex_calc = ai_fixture["complex_calculation"]
#     square = ai_fixture["square"]
#     expensive_operation = ai_fixture["expensive_operation"]

#     square_results = []
#     expensive_results = []

#     @ai.hook(square)
#     def capture_square(row: TestRow, output, x):
#         square_results.append((row.value, x, output))

#     @ai.hook(expensive_operation)
#     def capture_expensive(row: TestRow, output, x):
#         expensive_results.append((row.value, x, output))

#     @ai.dataset("test")
#     class TestDataset(Dataset[TestRow]): ...

#     dataset = TestDataset()
#     dataset.extend([TestRow(value=4)])

#     async def main(row: TestRow):
#         return await complex_calc(row.value)

#     ai.run(
#         main=main,
#         hooks=[capture_square, capture_expensive],
#         dataset=dataset,
#     )

#     assert square_results == [(4, 4, 16)]
#     assert expensive_results == [(4, 16, 32)]


@pytest.mark.asyncio
async def test_hooks_error_handling(ai: AI, simple_dataset_tasks):
    failing = simple_dataset_tasks["failing"]
    results = []

    @ai.hook(failing)
    def capture_error(row: TestRow, output, x):
        # Hook should not be called if task fails
        results.append((row.value, x))

    @ai.dataset("test")
    class TestDataset(Dataset[TestRow]): ...

    dataset = TestDataset()
    dataset.extend([TestRow(value=5)])

    async def main(row: TestRow):
        with pytest.raises(ValueError):
            await failing(row.value)
        return None

    ai.run(
        main=main,
        hooks=[capture_error],
        dataset=dataset,
    )

    assert len(results) == 0


@pytest.mark.asyncio
async def test_multiple_rows_with_hooks(ai, ai_fixture):
    square = ai_fixture["square"]
    results = []

    @ai.hook(square)
    def capture_result(row: CaptureRow, output, x):
        results.append(
            {
                "row_x": row.x,
                "expected": row.expected_output,
                "actual": output,
            }
        )

    @ai.dataset("test")
    class TestDataset(Dataset[CaptureRow]): ...

    dataset = TestDataset()
    dataset.extend(
        [
            CaptureRow(x=2, expected_output=4),
            CaptureRow(x=3, expected_output=9),
            CaptureRow(x=4, expected_output=16),
        ]
    )

    async def main(row: CaptureRow):
        return await square(row.x)

    ai.run(
        main=main,
        hooks=[capture_result],
        dataset=dataset,
    )

    assert len(results) == 3
    assert all(r["actual"] == r["expected"] for r in results)
    assert sorted([r["row_x"] for r in results]) == [2, 3, 4]
