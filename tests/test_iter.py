import pytest

from agentlens import gather, iterate, lens, provide, task


@task
@pytest.mark.asyncio
async def test_async_iteration_labeling():
    """Test that async tasks get properly labeled with iteration indices."""

    @task
    async def inner_task(idx: int):
        assert idx == lens.task.iteration
        return idx

    @task
    async def outer_task():
        tasks = [inner_task(i) for i in range(3)]
        return await gather(*tasks, desc="testing iteration")

    results = await outer_task()
    assert results == [0, 1, 2]


@task
@pytest.mark.asyncio
async def test_nested_async_iteration():
    """Test that nested async tasks maintain proper iteration indices."""
    results = []

    @task
    async def leaf_task(idx: int):
        assert idx == lens.task.iteration
        results.append(idx)

    @task
    async def mid_task(idx: int):
        assert idx == lens.task.iteration
        tasks = [leaf_task(i) for i in range(2)]
        await gather(*tasks, desc="inner tasks")

    @task
    async def root_task():
        tasks = [mid_task(i) for i in range(2)]
        await gather(*tasks, desc="outer tasks")

    await root_task()
    assert len(results) == 4  # 2 mid_tasks * 2 leaf_tasks


def test_sync_iteration():
    """Test that sync iteration properly sets iteration indices."""
    results = []

    @task
    async def record_iteration():
        results.append(lens.task.iteration)

    @task
    async def run_test():
        with provide():
            for i in iterate(range(3)):
                assert i == lens.task.iteration
                await record_iteration()  # Actually execute the task

    import asyncio

    asyncio.run(run_test())
    assert results == [0, 1, 2]


@task
@pytest.mark.asyncio
async def test_iteration_cleanup():
    """Test that iteration index is properly cleaned up after tasks complete."""

    @task
    async def check_iteration(expected_idx: int):
        assert lens.task.iteration == expected_idx

    @task
    async def parent_task():
        # Should reset between gather() calls
        tasks1 = [check_iteration(i) for i in range(2)]
        await gather(*tasks1)

        # Verify iteration is None after gather
        assert lens.task.iteration is None

        # Second batch should start fresh
        tasks2 = [check_iteration(i) for i in range(2)]
        await gather(*tasks2)

    await parent_task()


@task
@pytest.mark.asyncio
async def test_concurrent_iteration_isolation():
    """Test that concurrent tasks maintain isolated iteration indices."""
    results = []

    @task
    async def leaf_task():
        results.append(lens.task.iteration)

    @task
    async def branch_task(branch_idx: int):
        tasks = [leaf_task() for _ in range(2)]
        await gather(*tasks, desc=f"branch {branch_idx}")

    # Run two branches concurrently
    tasks = [branch_task(i) for i in range(2)]
    await gather(*tasks)

    # Each branch should have its leaf tasks numbered 0,1
    assert sorted(results) == [0, 0, 1, 1]
