from agentlens.dataset import Dataset, Label, Row, subset
from example.ai import ai


## define your row schema and pass it into a dataset
## this is just like a normal Pydantic model
## except it will not error on missing labels when you load the dataset
## it will only error if you try to access a missing label
## this allows you to progressively bootstrap Labels
## and everything is type-safe (Labels are assumed to exist by your IDE)
class InvoiceRow(Row):
    markdown: str
    total_cost: float = Label()
    contains_error: bool = Label()


# define dataset and pass it a name
@ai.dataset("invoices")
class InvoiceDataset(Dataset[InvoiceRow]): ...


# define some rows (note: don't have to add Labels yet)
row1 = InvoiceRow(markdown="invoice1...")
row2 = InvoiceRow(markdown="invoice2...")

# load the dataset
dataset = InvoiceDataset()

# adds rows, initializing the file if necessary
InvoiceDataset.extend([row1, row2])

# iterate over the rows
for row in dataset:
    print(row.markdown)  # type-safe

# access rows either via their numeric index or their string identifier
first_row = dataset[0]
specific_row = dataset["some_row_id"]

first_row.total_cost  # accessing a non-existent Label throws an error
first_row.total_cost = 100  # can mutate fields like this, e.g. to bootstrap a Label

dataset.save()  # saves the updated dataset, first ensuring it's in a valid state


# you can add subsets by providing functions that take a row
# and return a boolean indicating whether it should be included
# in the subset
# this functionality integrates with the task runner to
# organize and document your eval runs for you
@ai.dataset("invoices")
class InvoiceDataset(Dataset[InvoiceRow]):
    @subset()
    def september(self, row: InvoiceRow):
        return row.date_created.month == 9


# loads just the appropriate subset
september_invoices = InvoiceDataset("september")

# there are easy ways to migrate datasets by transitioning the data from your Columns to new Labels
# and then deleting the old Columns
# for this to work it also would need to pass through any unspecified fields, because those would be
# the previous data fields
