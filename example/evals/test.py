from pydantic import BaseModel

from agentlens import gather, generate_text, lens, task
from example.config import anthropic


@task
async def shit():
    await generate_text(
        model=anthropic / "claude-3-5-sonnet-20241022",
        system="You are a helpful assistant.",
        prompt="What is the capital of France?",
    )


class Test(BaseModel):
    a: int
    b: str


@task
async def shit2():
    tasks = [shit3(i) for i in range(3)]
    return await gather(*tasks, desc="doing shit")


@task
async def shit3(idx: int):
    assert idx == lens.task.iteration
    tasks = [shit4(i) for i in range(3)]
    return await gather(*tasks, desc="doing shit 3")


@task
async def shit4(idx: int):
    assert idx == lens.task.iteration
    return Test(a=idx, b=f"hello {idx}")
