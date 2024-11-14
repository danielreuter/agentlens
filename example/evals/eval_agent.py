import json
from dataclasses import dataclass

from example.agent import check_integrity, extract_total_cost, process_invoice
from example.config import ls
from example.datasets import InvoiceDataset, InvoiceExample

# hooks can be pre or post
# pre takes example, state, *args, **kwargs
# post takes example, state, output, *args, **kwargs


# an eval is a task that does:
# 1. sets up some data
# 2. sets up state
# 3. creates a runner that has access to the state, w/ hooks and caching logic
# 4. runs the runner in sync and async mode
# 5. aggregates the results into the eval directory
# 6. saves the dataset
# note: this is only implicitly an abstraction -- it's just a task!


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


@ls.eval()
def bootstrap_invoice_labels():
    dataset = InvoiceDataset("september")
    state = State(total_cost_diffs=[], correct_integrities=[])

    with ls.hooks(
        boot_check_integrity,
        boot_extract_total_cost,
    ):
        # force=False is default, which means it will throw an error if the context is already provided
        def eval(example):
            with ls.provide(state, example):
                return process_invoice(example.markdown)

        results = ls.gather(*[eval(example) for example in dataset], desc="Processing invoices")
        dataset.save()


@ls.eval()
def bootstrap_invoice_labels():
    dataset = InvoiceDataset("september")
    state = State(total_cost_diffs=[], correct_integrities=[])

    with ls.provide(
        state,
        hooks=[
            boot_check_integrity,
            boot_extract_total_cost,
        ],
    ):

        def eval(example):
            return process_invoice(example.markdown)

        dataset.run(eval)
        dataset.save()


def some_fn():
    example = InvoiceExample.get()
    with ls.context(example, override=True):
        pass


# label it as a task so that the "run" abstraction remains!


@dataclass
@ls.contextvar("state")  # type-safe context....
class State:
    total_cost_diffs: list[float]
    correct_integrities: list[bool]


class HookInfo: ...


class TaskParams: ...


InvoiceHookInfo = HookInfo[InvoiceExample, State]

ctx = ls.context()

openai = {}


def use(): ...


@ls.hook(check_integrity)
def hook_check_integrity0(params):
    params["model"] = openai / "o1-preview"
    output: str = yield params

    example = ls.get(InvoiceExample)
    state = ls.get(State)
    score = output == example.contains_error
    state.correct_integrities.append(score)

    ls.get(InvoiceExample)


@ls.hook(extract_total_cost, "post")
def hook_extract_total_cost0(example: InvoiceExample, state: State, output):
    score = output == example.total_cost
    state.total_cost_diffs.append(score)


@ls.hook(check_integrity)
def hook_check_integrity(example: InvoiceExample, state: State, output, *args, **kwargs):
    score = output == example.contains_error
    state.correct_integrities.append(score)


@ls.hook(extract_total_cost)
def hook_extract_total_cost(example: InvoiceExample, state: State, output, *args, **kwargs):
    score = output == example.total_cost
    state.total_cost_diffs.append(score)


@ls.eval()
def eval_process_invoice(subset):
    dataset = InvoiceDataset(subset)
    state = State(total_cost_diffs=[], correct_integrities=[])

    ls.info(dataset)
    ls.info(state)

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


class InvoiceEvalContext:
    invoice_example: InvoiceExample
    state: State


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

        (ls / "report.json").write_text(
            json.dumps(
                {
                    "total_cost_diffs": state.total_cost_diffs,
                    "correct_integrities": state.correct_integrities,
                    "results": results,
                }
            )
        )


# def inject_kwargs(example, state, *args, **kwargs):
#     kwargs["model"] = "o1-preview"
## alternative hook patterns -- meta hooks

# def read_cost(example, output):
#     example.total_cost = output

# def eval(example):
#     with ls.context(
#         example=example,
#         state=state,
#         hooks=[
#             hook_check_integrity,
#             hook_extract_total_cost_pre,
#             hook_extract_total_cost_post,
#             read_output(extract_total_cost, read_cost),
#             inject(extract_total_cost, model="openai:o1-preview"),
#         ],
#         cache=[check_integrity, extract_total_cost],
#     ):
#         return process_invoice(example.markdown)
#
