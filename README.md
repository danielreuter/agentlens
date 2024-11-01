# agentlens

This library contains a set of lightweight abstractions for building agent scaffolds that are easy to evaluate and maintain.

```bash
pip install agentlens
```

## Features

- **Decorator-driven logic**—define arbitrarily complex scaffolds and evaluations by composing functions
- **Expressive evaluation framework**—run evals with hooks for full control over your agent's computation graph
- **ORM for datasets**—quickly bootstrap type-safe, validated datasets with zero boilerplate
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
# File: /your_project/ai.py

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
            api_key="...",  # your Anthropic API key
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
- `cache: bool = False`--cache the input/output of the task for use in evaluations
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
async def transcribe_pdf(image: Image) -> str:
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
from agentlens import Dataset, Label, Row


class InvoiceRow(Row):
    markdown: str 
    date_created: datetime  
    total_cost: float = Label()  
    contains_error: bool = Label()  

# define dataset and give it a name
@ls.dataset("invoices")
class InvoiceDataset(Dataset[InvoiceRow]): 

    # define subsets as filters on the rows
    # the name is the argument passed to `subset`, defaulting to the function name
    @subset("september")
    def september(self, row: InvoiceRow):
        return row.date_created.month == 9


# create and add rows (labels can be added later)
row1 = InvoiceRow(markdown="invoice1...")
row2 = InvoiceRow(markdown="invoice2...")

# initialize dataset and add rows
dataset = InvoiceDataset("september")
InvoiceDataset.extend([row1, row2])

# access rows by index or ID
first_row = dataset[0]
specific_row = dataset["some_row_id"]

# labels are type-safe and validated
first_row.total_cost = 100  # set a label
print(first_row.total_cost)  # access a label (throws error if not set)

# save changes
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

The first thing we'll want to do is bootstrap targets for our `InvoiceDataset`. This is easy to do using the hooks system. 

We will use hooks to:
1. Modify the `check_integrity` and `extract_total_cost` tasks to use the `o1-preview` model, which is the most expensive and capable model available
2. Tap into the execution of these functions to write the results to the dataset as target labels

```python
dataset = InvoiceDataset("september")

@ls.hook(check_integrity, model="o1-preview")
def hook_check_integrity(row: InvoiceRow, output, *args, **kwargs):
    row.contains_error = not output

@ls.hook(extract_total_cost, model="o1-preview")
def hook_extract_total_cost(row: InvoiceRow, output, *args, **kwargs):
    row.total_cost = output

ls.run(
    main=process_invoice,
    dataset=dataset,
    hooks=[hook_check_integrity, hook_extract_total_cost],
)

dataset.save()
```

Now that we have labels, we can evaluate the `check_integrity` and `extract_total_cost` tasks as they were originally defined.

TODO: describe how to run evals from the CLI + the console UI / writing files

```python
@ls.task()
def eval_agent(subset: str | None = None):
    dataset = InvoiceDataset(subset)

    check_integrity_scores = []
    extract_total_cost_scores = []

    @ls.hook(check_integrity)
    def hook_check_integrity(row: InvoiceRow, output, *args, **kwargs):
        score = output == row.contains_error
        check_integrity_scores.append(score)

    @ls.hook(extract_total_cost)
    def hook_extract_total_cost(row: InvoiceRow, output, *args, **kwargs):
        score = output - row.total_cost
        extract_total_cost_scores.append(score)

    ls.run(
        dataset=dataset,
        hooks=[hook_check_integrity, hook_extract_total_cost],
        main=process_invoice,
    )

    ls.write_text(
        "report.md",
        f"""
        check_integrity (% correct): {sum(check_integrity_scores) / len(check_integrity_scores)}
        extract_total_cost (avg. error): {sum(extract_total_cost_scores) / len(extract_total_cost_scores)}
        """,
    )
```
