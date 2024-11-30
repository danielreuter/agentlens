import json

from pydantic import BaseModel

from agentlens import gather, generate_object, lens, task
from example.config import openai


@task
async def shit():
    response = await generate_object(
        model=openai / "gpt-4o-mini",
        schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "string"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        system="You are a helpful assistant.",
        prompt="What is the capital of France?",
    )
    (lens.task.dir / "response.json").write_text(json.dumps(response))
    return response


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
