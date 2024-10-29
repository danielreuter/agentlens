from example.agent import check_integrity, extract_total_cost, process_invoice
from example.ai import ai
from example.datasets import InvoiceDataset, InvoiceRow


def eval_agent(dataset: InvoiceDataset):
    @ai.hook(check_integrity, model="o1-preview")
    def hook_check_integrity(row: InvoiceRow, output, *args, **kwargs):
        row.contains_error = not output

    @ai.hook(extract_total_cost, model="o1-preview")
    def hook_extract_total_cost(row: InvoiceRow, output, *args, **kwargs):
        row.total_cost = output

    ai.run(
        main=process_invoice,
        dataset=dataset,
        hooks=[hook_check_integrity, hook_extract_total_cost],
    )

    dataset.save()


def eval_agent_with_report(dataset: InvoiceDataset):
    check_integrity_scores = []
    extract_total_cost_scores = []

    @ai.hook(check_integrity)
    def hook_check_integrity(input, output, row: InvoiceRow):
        score = output == row.contains_error
        check_integrity_scores.append(score)

    @ai.hook(extract_total_cost)
    def hook_extract_total_cost(input, output, row: InvoiceRow):
        score = output - row.total_cost
        extract_total_cost_scores.append(score)

    def report():
        return f"""
        check_integrity (% correct): {sum(check_integrity_scores) / len(check_integrity_scores)}
        extract_total_cost (avg. error): {sum(extract_total_cost_scores) / len(extract_total_cost_scores)}
        """

    ai.run(
        dataset=dataset,
        hooks=[hook_check_integrity, hook_extract_total_cost],
        main=process_invoice,
        report=report,
    )


def eval_fn(dataset: InvoiceDataset):
    check_integrity_scores = []
    extract_total_cost_scores = []

    @ai.hook(check_integrity)
    def hook_check_integrity(input, output, row: InvoiceRow):
        score = output == row.contains_error
        check_integrity_scores.append(score)

    @ai.hook(extract_total_cost)
    def hook_extract_total_cost(input, output, row: InvoiceRow):
        score = output - row.total_cost
        extract_total_cost_scores.append(score)

    def report():
        return f"""
        check_integrity (% correct): {sum(check_integrity_scores) / len(check_integrity_scores)}
        extract_total_cost (avg. error): {sum(extract_total_cost_scores) / len(extract_total_cost_scores)}
        """

    ai.run(
        dataset=dataset,
        hooks=[hook_check_integrity, hook_extract_total_cost],
        main=process_invoice,
        report=report,
    )


def eval_agent2(dataset: InvoiceDataset):
    @ai.hook(check_integrity, model="o1-preview")
    def hook_check_integrity(input, output, row: InvoiceRow):
        row.contains_error = not output

    @ai.hook(extract_total_cost, model="o1-preview")
    def hook_extract_total_cost(input, output, row: InvoiceRow):
        row.total_cost = output

    ai.run(
        main=process_invoice,
        dataset=dataset,
        hooks=[hook_check_integrity, hook_extract_total_cost],
    )

    dataset.save()
