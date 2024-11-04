from dataclasses import dataclass

from example.agent import check_integrity, extract_total_cost, process_invoice
from example.config import ls
from example.datasets import InvoiceDataset

# hooks can be pre or post
# pre takes example, state, *args, **kwargs
# post takes example, state, output, *args, **kwargs


@ls.hook(check_integrity, "pre")
def hook_check_integrity(example, state, *args, **kwargs):
    kwargs["model"] = "o1-preview"


@ls.hook(extract_total_cost, "pre")
def hook_extract_total_cost_pre(example, state, *args, **kwargs):
    kwargs["model"] = "o1-preview"


@ls.hook(extract_total_cost, "post")
def hook_extract_total_cost_post(example, state, output, *args, **kwargs):
    example.total_cost = output


# an eval is a task that does:
# 1. sets up some data
# 2. sets up state
# 3. creates a runner that has access to the state, w/ hooks and caching logic
# 4. runs the runner in sync and async mode
# 5. aggregates the results into the eval directory
# 6. saves the dataset
# note: this is only implicitly an abstraction -- it's just a task!


@dataclass
class EvalState:
    total_costs: list[float]


@ls.task()
async def eval_process_invoice_async(subset):
    dataset = InvoiceDataset(subset)
    state = EvalState(total_costs=[])

    async def eval(example):
        with ls.context(
            example=example,
            state=state,
            hooks=[
                hook_check_integrity,
                hook_extract_total_cost_pre,
                hook_extract_total_cost_post,
            ],
            cache=[check_integrity, extract_total_cost],
        ):
            return await process_invoice(example.markdown)

    results = await ls.gather(*[eval(example) for example in dataset])

    (ls / "report.md").write_text(f"""
        Total cost across all examples: {sum(state.total_costs)}  
        First example: {results[0]}
    """)

    dataset.save()


@ls.task()
def eval_process_invoice_sync(subset):
    dataset = InvoiceDataset(subset)
    state = {}

    def eval(example):
        with ls.context(
            example=example,
            state=state,
            hooks=[
                hook_check_integrity,
                hook_extract_total_cost_pre,
                hook_extract_total_cost_post,
            ],
            cache=[check_integrity, extract_total_cost],
        ):
            return process_invoice(example.markdown)

    results = [eval(example) for example in ls.iter(dataset)]

    (ls / "report.md").write_text(f"""
        Total cost across all examples: {sum(state.total_costs)}
        First example: {results[0]}
    """)

    dataset.save()


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
