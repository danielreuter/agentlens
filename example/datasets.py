from agentlens.dataset import Dataset, Label, Row, subset
from example.config import ls


class InvoiceRow(Row):
    markdown: str
    total_cost: float = Label()
    contains_error: bool = Label()


@ls.dataset("invoices")
class InvoiceDataset(Dataset[InvoiceRow]):
    @subset()
    def september(self, row: InvoiceRow):
        return row.date_created.month == 9


dataset = InvoiceDataset()
dataset.clear()
dataset.extend(
    [
        InvoiceRow(markdown="test1"),
        InvoiceRow(
            markdown="test2",
        ),
    ]
)
dataset.save()
