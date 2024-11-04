from agentlens.dataset import Dataset, Example, Label
from example.config import ls


class InvoiceExample(Example):
    markdown: str
    total_cost: float = Label()
    contains_error: bool = Label()


class InvoiceDataset(Dataset[InvoiceExample]):
    def __init__(self, subset: str | None):
        super().__init__(name="invoices", lens=ls, subset=subset)

    def filter(self, row: InvoiceExample):
        if self.subset == "september":
            return row.date_created.month == 9
        return True


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
