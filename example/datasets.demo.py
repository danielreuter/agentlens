from agentlens.dataset import Dataset, Example, Label
from example.config import ls


## define your row schema and pass it into a dataset
## this is just like a normal Pydantic model
## except it will not error on missing labels when you load the dataset
## it will only error if you try to access a missing label
## this allows you to progressively bootstrap Labels
## and everything is type-safe (Labels are assumed to exist by your IDE)
class InvoiceExample(Example):
    markdown: str
    date: str
    total_cost: float = Label()
    contains_error: bool = Label()


# define dataset and pass it a name
class InvoiceDataset(Dataset[InvoiceExample]):
    def __init__(self, subset: str | None = None):
        super().__init__(name="invoices", lens=ls, subset=subset)

    def filter(self, row: InvoiceExample):
        if self.subset == "september":
            return row.date_created.month == 9
        else:
            raise ValueError("Subset not implemented")


# define some rows (Labels can be added later)
example1 = InvoiceExample(markdown="invoice1...", date="2024-09-01")
example2 = InvoiceExample(markdown="invoice2...", date="2024-09-02")

# load the dataset
dataset = InvoiceDataset()

# adds rows, initializing the file if necessary
dataset.extend([example1, example2])

# iterate over the rows
for row in dataset:
    print(row.markdown)  # type-safe

# access rows by index or ID
first_example = dataset[0]
specific_example = dataset["some_example_id"]

# labels are type-safe and validated
first_example.total_cost = 100  # set a Label
print(first_example.total_cost)  # access a Label (throws error if not set)

# save changes, ensuring the dataset is in a valid state1
dataset.save()

# load a specific subset
september_invoices = InvoiceDataset("september")

# you can add subsets by providing functions that take a row
# and return a boolean indicating whether it should be included
# in the subset
# this functionality integrates with the task runner to
# organize and document your eval runs for you


# loads just the appropriate subset
september_invoices = InvoiceDataset("september")

# there are easy ways to migrate datasets by transitioning the data from your Columns to new Labels
# and then deleting the old Columns
# for this to work it also would need to pass through any unspecified fields, because those would be
# the previous data fields
