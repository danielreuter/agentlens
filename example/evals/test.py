from agentlens import lens, task
from agentlens.inference import generate_text
from example.config import openai


@task
async def shit():
    await generate_text(
        model=openai / "o1-preview",
        prompt="What is the capital of France?",
    )


@task
async def shit2():
    tasks = [shit3(i) for i in range(3)]
    await lens.gather(*tasks, desc="doing shit")


@task
async def shit3(idx: int):
    assert idx == lens.task.iteration
    tasks = [shit4(i) for i in range(3)]
    await lens.gather(*tasks, desc="doing shit 3")


@task
async def shit4(idx: int):
    assert idx == lens.task.iteration
