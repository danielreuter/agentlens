from agentlens.dataset import Row
from example.agent import check_integrity, extract_total_cost, process_invoice
from example.config import ls
from example.datasets import InvoiceDataset, InvoiceRow


@ls.task()
def bootstrap_labels(subset: str | None = None):
    dataset = InvoiceDataset(subset)

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

    dataset.save()


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

    async def main(row: Row):
        return await process_invoice(row.markdown)

    ls.run(
        main=main,
        dataset=dataset,
        hooks=[hook_check_integrity, hook_extract_total_cost],
    )

    ls.write_text(
        "report.md",
        f"""
        check_integrity (% correct): {sum(check_integrity_scores) / (len(check_integrity_scores)+1e-6)}
        extract_total_cost (avg. error): {sum(extract_total_cost_scores) / (len(extract_total_cost_scores)+1e-6)}
        """,
    )
