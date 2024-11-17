# agentlens

This library contains a set of lightweight abstractions for building agent scaffolds that are easy to evaluate and maintain.

```bash
pip install agentlens
```

## Overview
- [Configuration](#configuration)
- [Tasks](#tasks)
- [Inference](#inference)
- [Datasets](#datasets)
- [Evaluation](#evaluation)

## Tasks

The basic building block of the library is a **task** -- an asynchronous function decorated with `@task`. Tasks have access to customizable context variables via the `lens` object, whose state is initialized automatically when some top-level task is called in your application.

### Run

When tasks are called, they will automatically initialize a `Run` context if one has not already been initialized by a parent task. This contains metadata that groups together all of the tasks that are called within the same run.

If tasks are run from the CLI, then a new run directory with a unique ID will be created, and each task will be able to read and write files to this directory as well as to subdirectories nested in the same hierarchy in which the tasks are called.

e.g.:

```bash
agentlens run path/to/file.py task_name
```

Then tasks will be able to read and write files to the run directory, e.g.:

```python
import json

from agentlens import lens, task

@task
def write_report():
    (lens.run.dir / "report.md").write_text("...")


@task
def write_output():
    (lens.task.dir / "output.json").write_text(json.dumps({"result": 42}))
```

This provides a flexible way to observe the behavior of your agent as it runs. 

### Context

You can define context objects using the `@context` decorator and provide them to all nested tasks like so:

```python
from agentlens import context, task, lens, provide


@context
class State:
    some_value: int


@task
async def main_task():
    state = State(some_value=1)

    with provide(state):
        result = await nested_task()
        return result + 1  # 2


@task
async def nested_task():
    state = lens[State]  # access the State object, erroring if none is provided
    return state.some_value  # 1
```

These context variables are async-friendly and thread-safe. They will only be accessible by code that is nested within the context manager that provides them.

You can delete or set context variables like so:

```python
@task
async def nested_task():
    del lens[State]  # delete the State object
    lens[State] = State(some_value=2)  # set the State object
```

Todo: describe how this is implemented using contextvars...


### Hooks

Hooks are functions that can modify the behavior of tasks. They can take any of the tasks arguments and can modify the arguments that the task is called with by passing in a dictionary of keyword arguments to the `yield` statement:

```python
from agentlens import task, provide, lens
import agentlens.evaluation as ev

@ev.hook(some_task)
def hook_some_task(a: int) -> ev.GeneratorHook[int]:
    state = lens[State]
    state.a += 1
    task_output = yield {"a": a + 1}  # modify the arguments that the task is called with
    state.a += 1


@task
async def eval_some_task():
    state = State(a=1)
    hooks = [hook_some_task]

    with provide(state, hooks=hooks):
        return await some_task()

    (lens / "report.md").write_text(f"a: {state.a}")
```

### Mocks

Often you will want to run a particular slice of the graph in isolation
to do this, you can attach mock functions to your tasks, which should adopt the
same interface as the task itself -- potentially reading from your database or local
filesystem to provide the necessary data, instead of calling expensive third-party APIs
like inference providers.

This API makes no assumptions about how your backend works


The arguments and return types must match that of the target function (the
arguments can be a subset of the target function's arguments)
This will be checked at runtime, similar to how hooks are validated.
This behavior can be turned off in the Lens config with `strict=False`.

Mocks are functions that can be used to replace the behavior of other functions for the purpose of testing. They are defined using the `@mock` decorator, and they can be used to replace the behavior of other functions by passing in a `replace` argument to the `provide` context manager.

```python
import agentlens.evaluation as ev


@ev.mock(some_task)
def mock_some_task(a: int) -> int:
    return a + 1


@task
async def eval_some_task():
    mocks = [mock_some_task]
    with provide(mocks=mocks):
        return await some_task(1)  # 2
```

This will replace the behavior of `some_task` with that of the mock function.

If you want finer-grained control over mocking behavior--e.g. if you are running a certain task in a loop and only want to mock the first few calls--you can simply delete the mock from the `lens` object, and since this uses contextvars, it will only affect tasks nested below.

```python
@task
async def eval_some_task():
    with provide(mocks=[mock_some_task]):
        for i in range(10):
            if i < 3:
                with remove(mock_some_task):
                    await some_task(1)  # 1
            else:
                await some_task(1)  # 2
    ```

## Inference

The library exposes a boilerplate-free wrapper around the OpenAI and Anthropic APIs. 

In the simplest case, you might just want to feed some model a user prompt and (optionally) a system prompt, and have it return a string using `generate_text`:

```python
from agentlens import generate_text, OpenAI, Anthropic


anthropic = Anthropic(
    api_key="...",
    max_connections_default=10,
    max_connections={
        "claude-3-5-sonnet": 30,
    },
)

openai = OpenAI(
    api_key="...",
    max_connections_default=10,
    max_connections={
        "gpt-4o-mini": 30,
    },
)


@task
async def summarize(text: str) -> str:
    return await generate_text(
        model=anthropic / "claude-3-5-sonnet",
        system="You are a helpful assistant.",
        prompt=f"""
            Please summarize the following text:

            {text}
            """,
        dedent=True,  # defaults to True, eliminating indents from all prompts using textwrap.dedent
        max_retries=3,  # number of retries on failure, defaults to 3
    )
```

To phrase more complex requests, you may opt to pass the model a list of messages:

```python
from PIL import Image


@task
async def transcribe_pdf(image: Image.Image) -> str:
    return await generate_text(
        model=openai / "gpt-4o-mini",
        messages=[
            ai.message.system("You are a helpful assistant."),
            ai.message.user(
                "Please transcribe the following PDF page to Markdown:",
                ai.message.image(image),
            ),
        ],
    )
```
> If you pass a `messages` argument, an exception will be raised if you also pass a `system` or `prompt` argument.

To request a structured output from the model, you can use `generate_object` and pass a Pydantic model as the `schema` argument.  

```python
from pydantic import BaseModel


class PDFMetadata(BaseModel):
    title: str | None
    author: str | None


@task
async def extract_pdf_metadata(image: Image) -> PDFMetadata:
    return await generate_object(
        model=openai / "gpt-4o",
        schema=PDFMetadata,
        messages=[
            ai.message.system("You are a helpful assistant."),
            ai.message.user(
                "Extract metadata from the following PDF page:",
                ai.message.image(image),
            ),
        ],
    )
```