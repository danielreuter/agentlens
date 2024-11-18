import json
from dataclasses import dataclass

import agentlens.evaluation as ev
from agentlens import gather, lens, provide, task
from example.agent import check_integrity, extract_total_cost, process_invoice
from example.config import openai
from example.datasets import InvoiceDataset, InvoiceExample

# Inference provider


# simple sync eval
# - demonstrates `lens.iter`
@task
def run_process_invoice_simple_sync():
    dataset = InvoiceDataset("september")

    for invoice in lens.iter(dataset, desc="Processing invoices"):
        process_invoice(invoice)


# simple async eval
# - demonstrates `gather`
@task
async def run_process_invoice_simple_async():
    dataset = InvoiceDataset("september")

    async def run(invoice):
        return await process_invoice(invoice)

    tasks = [run(invoice) for invoice in dataset]
    await gather(*tasks, desc="Processing invoices")


# simple context
# - demonstrates `lens.provide`
# - demonstrates `lens.__getitem__`
# Takeaways:
# - you can use AgentLens to create type-safe contextvars
# - namespacing is provided by the class
@dataclass
class Scores:
    total_cost_diffs: list[float]
    correct_integrities: list[bool]


@task
async def some_task_creating_context() -> int:
    scores = Scores(total_cost_diffs=[], correct_integrities=[])
    with provide(scores):
        test = await some_task_using_context(1.0)
        return test


@task
async def some_task_using_context(num: float) -> int:
    scores = lens[Scores]
    scores.total_cost_diffs.append(num)
    return 3


# simple hooks -- bootstrapping
# - demonstrates `lens.hook`
# - demonstrates `lens.hooks`
@ev.hook(check_integrity)
def boot_check_integrity() -> ev.HookGenerator[bool]:
    example = lens[InvoiceExample]
    result = yield {"model": openai / "o1-preview"}
    example.contains_error = result


@ev.hook(extract_total_cost)
def boot_extract_total_cost() -> ev.HookGenerator[float]:
    example = lens[InvoiceExample]
    result = yield {"model": openai / "o1-preview"}
    example.total_cost = result


@task
async def bootstrap_invoice_labels():
    dataset = InvoiceDataset("september")
    hooks = [
        boot_check_integrity,
        boot_extract_total_cost,
    ]
    mocks = []

    async def eval(example):
        with provide(example, hooks=hooks, mocks=mocks):
            return await process_invoice(example.markdown)

    tasks = [eval(example) for example in dataset]
    await gather(*tasks, desc="Processing invoices")

    dataset.save()


## note-- above shows how each eval can exert fine-grained control over the computation graph
## without polluting the callenstack


## now let's see how we can use these hooks to collect metrics in an eval


@ev.hook(check_integrity)
def hook_check_integrity() -> ev.HookGenerator[str]:
    example, scores = lens[InvoiceExample], lens[Scores]
    output = yield {}
    score = output == example.contains_error
    scores.correct_integrities.append(score)


@ev.hook(extract_total_cost)
def hook_extract_total_cost() -> ev.HookGenerator[float]:
    example, scores = lens[InvoiceExample], lens[Scores]
    result = yield {}
    scores.total_cost_diffs.append(result == example.total_cost)


@task
async def eval_process_invoice():
    dataset = InvoiceDataset("september")
    scores = Scores(total_cost_diffs=[], correct_integrities=[])
    hooks = [
        hook_check_integrity,
        hook_extract_total_cost,
    ]

    async def eval(example):
        with provide(scores, example, hooks=hooks):
            return await process_invoice(example.markdown)

    tasks = [eval(example) for example in dataset]
    results = await gather(*tasks, desc="Processing invoices")

    (lens / "report.json").write_text(
        json.dumps(
            {
                "results": results,
                "accuracy": {
                    "check_integrity": mean(scores.correct_integrities),
                    "extract_total_cost": mean(scores.total_cost_diffs),
                },
            }
        )
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if len(values) > 0 else 0.0
