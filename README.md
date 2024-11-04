# agentlens

This library contains a set of lightweight abstractions for building agent scaffolds that are easy to evaluate and maintain.

```bash
pip install agentlens
```

## Features

- **Decorator-driven logic**—define arbitrarily complex scaffolds and evaluations by composing functions
- **Expressive evaluation framework**—run evals with hooks for full control over your agent's computation graph
- **Type-safe datasets**—bootstrap type-safe, validated datasets with zero boilerplate
- **Built-in observability**—easy integration with Langfuse
- **Clean inference API**—call models using a syntax inspired by Vercel's very elegant [AI SDK](https://sdk.vercel.ai/docs/introduction)

## Overview
- [Configuration](#configuration)
- [Tasks](#tasks)
- [Inference](#inference)
- [Datasets](#datasets)
- [Evaluation](#evaluation)

## Configuration

Initialize a `Lens` object to manage your project's observability and evaluation logic, and an `AI` object for clean access to OpenAI and Anthropic models.

```python
# File: /your_project/config.py

import os
from pathlib import Path

from dotenv import load_dotenv

from agentlens import AI, Lens, OpenAIProvider, AnthropicProvider

load_dotenv()

ROOT_DIR = Path(__file__).parent

ls = Lens(
    runs_dir=ROOT_DIR / "runs",  # where to store runs
    dataset_dir=ROOT_DIR / "datasets",  # where to store datasets
)

ai = AI(
    providers=[
        OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            max_connections={ # global concurrency limits set on a per-model basis
                "DEFAULT": 10,
                "o1-preview": 2,
                "gpt-4o-mini": 30,
            },
        ),
        AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            max_connections={
                "DEFAULT": 10,
                "claude-3-5-sonnet": 5,
            },
        ),
    ],
)
```
By default API keys will be read from environment variables, but you can also pass them in directly.

## Tasks

The basic building block of the library is a **task**. A task is a function that makes one or more calls to an AI model. 

Declaring a function as a task enters it into a unified observability and evaluation ecosystem. Do so using the `task` decorator on the `Lens` object:

```python
from your_project.config import ls


@ls.task()
def some_task(some_input: str) -> str:
    pass  # insert some AI logic here
```

The `task` decorator takes the following optional arguments:
- `name: str | None = None`--a name for the task, which will be used in the UI and in logging
- `max_retries: int = 0`--number of retries on failure, defaults to 0

## Inference

The library exposes a boilerplate-free wrapper around the OpenAI and Anthropic APIs. 

In the simplest case, you might just want to feed some model a user prompt and (optionally) a system prompt, and have it return a string using `generate_text`:

```python
from your_project.config import ai, ls


@ls.task()
async def summarize(text: str) -> str:
    return await ai.generate_text(
        model="gpt-4o-mini",
        system="You are a helpful assistant.",
        prompt=f"""
            Please summarize the following text:

            {text}
            """,
        dedent=True,  # defaults to True, eliminating indents from all prompts using textwrap.dedent
        max_attempts=3,  # number of retries on failure, defaults to 3
    )
```

To phrase more complex requests, you may opt to pass the model a list of messages:

```python
from PIL import Image
from your_project.config import ai, ls


@ls.task()
async def transcribe_pdf(image: Image.Image) -> str:
    return await ai.generate_text(
        model="gpt-4o-mini",
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

To request a structured output from the model, you can use `generate_object` and pass a Pydantic model as the `type` argument.  

```python
class PDFMetadata(BaseModel):
    title: str | None
    author: str | None


@ls.task()
async def extract_pdf_metadata(image: Image) -> PDFMetadata:
    return await ai.generate_object(
        model="gpt-4o",
        type=PDFMetadata,
        messages=[
            ai.message.system("You are a helpful assistant."),
            ai.message.user(
                "Extract metadata from the following PDF page:",
                ai.message.image(image),
            ),
        ],
    )
```

## Datasets

The library exposes an ORM-like API for developing evaluation datasets. 

A `Dataset` is defined by a `Row` schema and a name. The name will identify it in the datasets directory, as well as in the UI and in eval logs.

`Row` is just like a normal Pydantic model, except it will not error on missing labels when you load the dataset -- it will only error if you try to access a missing label, e.g. in a hook. This allows you to progressively bootstrap type-safe labels. 

```python
from datetime import datetime
from agentlens import Dataset, Example, Label


class InvoiceExample(Example):
    markdown: str
    date: str
    total_cost: float = Label()
    contains_error: bool = Label()


# define dataset and pass it a name
class InvoiceDataset(Dataset[InvoiceExample]):
    def __init__(self, subset: str | None = None):
        super().__init__(name="invoices", lens=ls, subset=subset)

    def filter(self, row: InvoiceExample):
        if self.subset == "september":
            return row.date_created.month == 9
        else:
            raise ValueError("Subset not implemented")


# define some rows (Labels can be added later)
example1 = InvoiceExample(markdown="invoice1...", date="2024-09-01")
example2 = InvoiceExample(markdown="invoice2...", date="2024-09-02")

# load the dataset
dataset = InvoiceDataset()

# adds rows, initializing the file if necessary
dataset.extend([example1, example2])

# iterate over the rows
for row in dataset:
    print(row.markdown)  # type-safe

# access rows by index or ID
first_example = dataset[0]
specific_example = dataset["some_example_id"]

# labels are type-safe and validated
first_example.total_cost = 100  # set a Label
print(first_example.total_cost)  # access a Label (throws error if not set)

# save changes, ensuring the dataset is in a valid state1
dataset.save()

# load a specific subset
september_invoices = InvoiceDataset("september")
```

## Evaluation

The evaluation API uses hooks to give you fine-grained control over your agent's computation graph, and you can run them either from a Jupyter cell or from the CLI. 

First let's define a simple set of tasks, riffing off of the invoice data structure we defined in the `Dataset` section:

```python
@ls.task()
async def process_invoice(invoice: str) -> float | str:
    looks_fine = await check_integrity(invoice)

    if not looks_fine:
        return await generate_error_report(invoice)

    return await extract_total_cost(invoice)


@ls.task()
async def check_integrity(invoice: str, model: str = "gpt-4o-mini") -> bool:
    return await ai.generate_object(
        model=model,
        type=bool,
        prompt=f"Return True if the invoice looks uncorrupted: {invoice.text}",
    )


@ls.task()
async def generate_error_report(invoice: str) -> str:
    return await ai.generate_text(
        model="gpt-4o",
        prompt=f"Write an error report for this corrupted invoice: {invoice.text}",
    )


@ls.task()
async def extract_total_cost(invoice: str, model: str = "gpt-4o") -> float:
    return await ai.generate_object(
        model=model,
        type=float,
        prompt=f"Extract the total cost from this invoice: {invoice.text}",
    )
```

The first thing we'll want to do is bootstrap labels for our `InvoiceDataset`. This is easy to do using the hooks system. 

We will use hooks to:
1. Modify the `check_integrity` and `extract_total_cost` tasks to use the `o1-preview` model, which is the most expensive and capable model available
2. Write the results to the dataset as target labels

```python
@ls.hook(check_integrity, "wrap")
def boot_check_integrity(example, state, *args, **kwargs):
    kwargs["model"] = "o1-preview"
    output = yield args, kwargs
    example.contains_error = not output


@ls.hook(extract_total_cost, "wrap")
def boot_extract_total_cost(example, state, *args, **kwargs):
    kwargs["model"] = "o1-preview"
    output = yield args, kwargs
    example.total_cost = output


@ls.task()
def bootstrap_invoice_labels():
    dataset = InvoiceDataset("september")
    with ls.context(
        hooks=[
            boot_check_integrity,
            boot_extract_total_cost,
        ]
    ):

        def eval(example):
            return process_invoice(example.markdown)

        dataset.run(eval)
        dataset.save()

bootstrap_invoice_labels()
```

Now that we have labels, we can evaluate the `check_integrity` and `extract_total_cost` tasks as they were originally defined.

TODO: describe how to run evals from the CLI + the console UI / writing files

```python

@dataclass
class State:
    total_cost_diffs: list[float]
    correct_integrities: list[bool]


@ls.hook(check_integrity, "post")
def hook_check_integrity(example, state, output, *args, **kwargs):
    score = output == example.contains_error
    state.correct_integrities.append(score)


@ls.hook(extract_total_cost, "post")
def hook_extract_total_cost(example, state, output, *args, **kwargs):
    score = output == example.total_cost
    state.total_cost_diffs.append(score)


def eval_process_invoice(subset):
    dataset = InvoiceDataset(subset)
    state = State(total_cost_diffs=[], correct_integrities=[])

    with ls.context(
        state=state,
        hooks=[
            hook_check_integrity,
            hook_extract_total_cost,
        ],
    ):

        def eval(example):
            return process_invoice(example.markdown)

        results = dataset.run(eval)

        (ls / "report.md").write_text(f"""
            Average total cost error: {sum(state.total_cost_diffs) / len(state.total_cost_diffs)}  
            Percent integrity correct: {sum(state.correct_integrities) / len(state.correct_integrities)}
            First result: {results[0]}
        """)
```

This process is I/O bound, so we'll want to run it asynchronously. The library maxes out your call rate on a per-model basis and prioritizes completing tasks in order. 

```python
@ls.task()
async def eval_process_invoice_async(subset):
    dataset = InvoiceDataset(subset)
    state = State(total_cost_diffs=[], correct_integrities=[])

    async with ls.eval_context(
        state=state,
        hooks=[
            hook_check_integrity,
            hook_extract_total_cost,
        ],
        cache=[check_integrity, extract_total_cost],
    ):

        async def eval(example):
            return await process_invoice(example.markdown)

        results = await dataset.run(eval)

        (ls / "report.md").write_text(f"""
            Average total cost error: {sum(state.total_cost_diffs) / len(state.total_cost_diffs)}  
            Percent integrity correct: {sum(state.correct_integrities) / len(state.correct_integrities)}
            First result: {results[0]}
        """)
```