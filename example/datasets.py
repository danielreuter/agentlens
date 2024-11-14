from agentlens.dataset import Dataset, Example, Label
from example.config import ls


class InvoiceExample(Example):
    markdown: str
    date_created: str
    total_cost: float = Label()
    contains_error: bool = Label()


# this should just take a dataset name and a dataset_dir, no need for lens
class InvoiceDataset(Dataset[InvoiceExample]):
    def __init__(self, subset: str | None = None):
        super().__init__("invoices", ls, subset)

    def filter(self, row: InvoiceExample):
        if self.subset == "september":
            return row.date_created == "2024-09-01"
        return True


dataset = InvoiceDataset()
dataset.clear()
dataset.extend(
    [
        InvoiceExample(markdown="test1", date_created="2024-09-01"),
        InvoiceExample(markdown="test2", date_created="2024-10-01"),
    ]
)
dataset.save()
