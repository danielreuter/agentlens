import json
from dataclasses import dataclass

from agentlens.hooks import GeneratorHook
from example.agent import check_integrity, extract_total_cost, process_invoice
from example.config import ls
from example.datasets import InvoiceDataset, InvoiceExample


# Inference provider
class OpenAI:
    def __truediv__(self, model: str) -> str:
        return model

    def __getitem__(self, model: str) -> str:
        return model


openai = OpenAI()


# simple sync eval
# - demonstrates `ls.iter`
@ls.eval()
def run_process_invoice_simple_sync():
    dataset = InvoiceDataset("september")

    for invoice in ls.iter(dataset, desc="Processing invoices"):
        process_invoice(invoice)


# simple async eval
# - demonstrates `ls.gather`
@ls.eval()
async def run_process_invoice_simple_async():
    dataset = InvoiceDataset("september")

    async def run(invoice):
        return await process_invoice(invoice)

    tasks = [run(invoice) for invoice in dataset]
    ls.gather(*tasks, desc="Processing invoices")


# simple context
# - demonstrates `ls.provide`
# - demonstrates `ls.__getitem__`
# Takeaways:
# - you can use AgentLens to create type-safe contextvars
# - namespacing is provided by the class
@dataclass
class Scores:
    total_cost_diffs: list[float]
    correct_integrities: list[bool]


@ls.task()
async def some_task_creating_context() -> int:
    scores = Scores(total_cost_diffs=[], correct_integrities=[])
    with ls.provide(scores):
        test = await some_task_using_context(1.0)
        return test


@ls.task()
async def some_task_using_context(num: float) -> int:
    scores = ls[Scores]
    scores.total_cost_diffs.append(num)
    return 3


# simple hooks -- bootstrapping
# - demonstrates `ls.hook`
# - demonstrates `ls.hooks`
@ls.hook(check_integrity)
def boot_check_integrity() -> GeneratorHook[bool]:
    example = ls[InvoiceExample]
    result = yield {"model": openai / "o1-preview"}
    example.contains_error = result


@ls.hook(extract_total_cost)
def boot_extract_total_cost() -> GeneratorHook[float]:
    example = ls[InvoiceExample]
    result = yield {"model": openai / "o1-preview"}
    example.total_cost = result


@ls.eval()
async def bootstrap_invoice_labels():
    dataset = InvoiceDataset("september")
    hooks = [
        boot_check_integrity,
        boot_extract_total_cost,
    ]

    async def eval(example):
        with ls.provide(example, hooks=hooks):
            return await process_invoice(example.markdown)

    tasks = [eval(example) for example in dataset]
    ls.gather(*tasks, desc="Processing invoices")

    dataset.save()


## note-- above shows how each eval can exert fine-grained control over the computation graph
## without polluting the callstack


## now let's see how we can use these hooks to collect metrics in an eval


@ls.hook(check_integrity)
def hook_check_integrity() -> GeneratorHook[str]:
    example, scores = ls[InvoiceExample], ls[Scores]
    output = yield {}
    score = output == example.contains_error
    scores.correct_integrities.append(score)


@ls.hook(extract_total_cost)
def hook_extract_total_cost() -> GeneratorHook[float]:
    example, scores = ls[InvoiceExample], ls[Scores]
    result = yield {}
    scores.total_cost_diffs.append(result == example.total_cost)


@ls.eval()
async def eval_process_invoice():
    dataset = InvoiceDataset("september")
    scores = Scores(total_cost_diffs=[], correct_integrities=[])
    hooks = [
        hook_check_integrity,
        hook_extract_total_cost,
    ]

    async def eval(example):
        with ls.provide(scores, example, hooks=hooks):
            return await process_invoice(example.markdown)

    tasks = [eval(example) for example in dataset]
    results = ls.gather(*tasks, desc="Processing invoices")

    (ls / "report.json").write_text(
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


# often you will want to run a particular slice of the graph in isolation
# to do this, you can attach mock functions to your tasks, which should adopt the
# same interface as the task itself -- potentially reading from your database or local
# filesystem to provide the necessary data, instead of calling expensive third-party APIs
# like inference providers.

# this API makes no assumptions about how your backend works


async def mock_some_task(arg: bool) -> bool:
    return False


@ls.task(mock=mock_some_task)
async def some_task(arg: bool) -> bool:
    return not arg


# the arguments and return types must match that of the target function (the
# arguments can be a subset of the target function's arguments)
# this will be checked at runtime, similar to how hooks are validated.
# This behavior can be turned off in the Lens config with strict=False
