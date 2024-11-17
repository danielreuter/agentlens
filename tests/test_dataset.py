# from datetime import datetime

# import pytest
# from pytest import fixture

# from agentlens.dataset import Dataset, Example, Label


# class InvoiceRow(Example):
#     markdown: str
#     date_created: datetime
#     total_cost: float = Label()
#     contains_error: bool = Label()


# @fixture
# def InvoiceDataset(ls):
#     class InvoiceDataset(Dataset[InvoiceRow]):
#         def __init__(self, subset: str | None = None):
#             super().__init__(name="invoices", lens=ls, subset=subset)

#         def filter(self, row: InvoiceRow) -> bool:
#             if self.subset == "errors":
#                 return row.contains_error
#             return True

#     return InvoiceDataset


# def test_dataset_initialization(InvoiceDataset):
#     """Test that a new dataset initializes correctly with no rows."""
#     dataset = InvoiceDataset()
#     assert len(dataset) == 0


# def test_label_validation(InvoiceDataset):
#     """Test that Label fields can be missing, None, or have values."""
#     # Test missing fields
#     row1 = InvoiceRow(markdown="invoice1...", date_created=datetime.now())

#     # Test explicit None
#     row2 = InvoiceRow(markdown="invoice2...", date_created=datetime.now())

#     # Test with values
#     row3 = InvoiceRow(
#         markdown="invoice3...", date_created=datetime.now(), total_cost=100.0, contains_error=True
#     )

#     dataset = InvoiceDataset()
#     dataset.extend([row1, row2, row3])

#     # Check missing fields raise appropriate errors
#     with pytest.raises(AttributeError):
#         _ = row1.total_cost

#     # Check None values
#     with pytest.raises(AttributeError):
#         _ = row2.total_cost

#     # Check actual values
#     assert row3.total_cost == 100.0
#     assert row3.contains_error is True


# def test_dataset_append(InvoiceDataset):
#     """Test that rows can be appended to the dataset."""
#     row1 = InvoiceRow(markdown="invoice1...", date_created=datetime.now())
#     row2 = InvoiceRow(markdown="invoice2...", date_created=datetime.now())

#     dataset = InvoiceDataset()
#     dataset.extend([row1, row2])

#     assert len(dataset) == 2
#     assert dataset[0].markdown == "invoice1..."
#     assert dataset[1].markdown == "invoice2..."


# def test_access_by_index(InvoiceDataset):
#     """Test accessing rows by index."""
#     row1 = InvoiceRow(markdown="invoice1...", date_created=datetime.now())

#     dataset = InvoiceDataset(None)
#     dataset.extend([row1])

#     retrieved_row = dataset[0]
#     assert retrieved_row.markdown == "invoice1..."


# def test_setting_label(InvoiceDataset):
#     """Test that labels can be set on a row."""
#     row1 = InvoiceRow(markdown="invoice1...", date_created=datetime.now())
#     row1.total_cost = 100.0
#     assert row1.total_cost == 100.0


# def test_saving_and_loading(InvoiceDataset):
#     """Test saving a dataset to disk and loading it back."""
#     row1 = InvoiceRow(markdown="invoice1...", date_created=datetime.now())

#     dataset = InvoiceDataset()
#     dataset.extend([row1])
#     dataset.save()

#     loaded_dataset = InvoiceDataset()
#     assert len(loaded_dataset) == 1
#     assert loaded_dataset[0].markdown == "invoice1..."


# def test_subsets(InvoiceDataset):
#     """Test that subsets of the dataset are correctly filtered."""
#     row1 = InvoiceRow(markdown="invoice1...", date_created=datetime.now(), contains_error=True)
#     row2 = InvoiceRow(markdown="invoice2...", date_created=datetime.now(), contains_error=False)

#     dataset = InvoiceDataset()
#     dataset.extend([row1, row2])
#     dataset.save()
