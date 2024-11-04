from agentlens.dataset import Dataset, Example, Label, subset
from example.config import ls


class InvoiceExample(Example):
    markdown: str
    total_cost: float = Label()
    contains_error: bool = Label()


@ls.dataset("invoices")
class InvoiceDataset(Dataset[InvoiceExample]):
    @subset()
    def september(self, row: InvoiceExample):
        return row.date_created.month == 9


dataset = InvoiceDataset()
dataset.clear()
dataset.extend(
    [
        InvoiceExample(markdown="test1"),
        InvoiceExample(
            markdown="test2",
        ),
    ]
)
dataset.save()
