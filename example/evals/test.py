# %%

from example.agent import check_integrity, extract_total_cost, process_invoice
from example.config import ls
from example.datasets import InvoiceDataset, InvoiceRow


@ls.task()
def bootstrap_labels(subset: str | None = None):
    dataset = InvoiceDataset(subset)
    print("len", len(dataset))

    @ls.hook(check_integrity, model="o1-preview")
    def hook_check_integrity(row: InvoiceRow, output, *args, **kwargs):
        row.contains_error = not output

    @ls.hook(extract_total_cost, model="o1-preview")
    def hook_extract_total_cost(row: InvoiceRow, output, *args, **kwargs):
        row.total_cost = output

    async def main(row: InvoiceRow):
        return await process_invoice(row.markdown)

    ls.run(
        main=main,
        dataset=dataset,
        hooks=[hook_check_integrity, hook_extract_total_cost],
    )


# %%
bootstrap_labels()

# %%
