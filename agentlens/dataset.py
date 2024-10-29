from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    Iterator,
    Type,
    TypeVar,
    Union,
)

from pydantic import (
    BaseModel,
    Field,
)

from agentlens.exceptions import DatasetIndexError
from agentlens.utils import create_readable_id

MISSING_LABEL = object()
SUBSET_FILTER_FN_INDICATOR = "_is_subset_filter_function"


def Label(
    default: Any = MISSING_LABEL,
    **kwargs,
) -> Field:
    return Field(default=default, **kwargs)


class Row(BaseModel):
    """
    A row in a Dataset. Will not raise validation errors if fields designated
    as Labels are missing
    """

    id: str = Field(default_factory=create_readable_id)

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if attr is MISSING_LABEL:
            raise AttributeError(f"Missing label: {name}")
        return attr

    # TODO: implement __setattr__ with validation at set-time instead of save-time


RowT = TypeVar("RowT", bound=Row)


class DatasetMetadata(BaseModel):
    version: int
    created_at: datetime
    last_modified: datetime


class DatasetFile(BaseModel, Generic[RowT]):
    metadata: DatasetMetadata
    rows: list[RowT]


class Dataset(Generic[RowT]):
    name: ClassVar[str]
    dataset_dir: ClassVar[Path]
    subset: str | None
    rows: list[RowT]
    row_type: ClassVar[Type[RowT]] = Row
    _file: DatasetFile[RowT]

    def __init__(self, subset: str | None = None):
        self.subset = subset
        self._file = self._read_file()

        rows = self._file.rows
        if subset:
            subset_rows, _ = self.split_rows(subset, rows)
            self.rows = subset_rows
        else:
            self.rows = rows

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in cls.__orig_bases__:
            if getattr(base, "__origin__", None) is Dataset:
                args = base.__args__
                if args:
                    cls.row_type = args[0]
                    break

    @property
    def file_path(self) -> Path:
        # stub = f".{self.subset}" if self.subset else ""
        stub = ""
        path = self.dataset_dir / f"{self.name}{stub}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _read_file(self) -> DatasetFile[RowT]:
        if not self.file_path.exists():
            return DatasetFile(
                metadata=DatasetMetadata(
                    version=1,
                    created_at=datetime.now(),
                    last_modified=datetime.now(),
                ),
                rows=[],
            )
        json_str = self.file_path.read_text()
        return DatasetFile[self.row_type].model_validate_json(json_str)

    def split_rows(self, subset: str, rows: list[RowT]) -> tuple[list[RowT], list[RowT]]:
        if not hasattr(self, subset):
            raise ValueError(f"Subset filter '{subset}' not found")

        filter_method = getattr(self, subset)
        if not hasattr(filter_method, SUBSET_FILTER_FN_INDICATOR):
            raise ValueError(f"'{subset}' is not a subset filter")

        subset_rows = []
        other_rows = []

        for row in rows:
            if filter_method(row):
                subset_rows.append(row)
            else:
                other_rows.append(row)

        return subset_rows, other_rows

    def extend(self, rows: list[RowT]) -> None:
        """Appends rows to the dataset in-place"""
        self.rows.extend(rows)

    def clear(self) -> None:
        """Clears the dataset in-place"""
        self.rows.clear()

    def save(self) -> None:
        """
        Saves the current version of the subset to disk, leaving other rows unchanged.
        """
        # concatenate current subset's rows with the rest of the dataset's rows
        new_rows = self.rows
        if self.subset:
            _, other_rows = self.split_rows(self.subset, self._file.rows)
            new_rows.extend(other_rows)

        # overwrite current dataset with new rows
        self._file.rows = new_rows
        json_str = self._file.model_dump_json(indent=2, exclude_defaults=True)
        self.file_path.write_text(json_str)

    def __iter__(self) -> Iterator[RowT]:
        return iter(self.rows)

    def __getitem__(self, key: Union[int, str, Any]) -> RowT:
        if isinstance(key, int):
            if 0 <= key < len(self.rows):
                return self.rows[key]
            raise DatasetIndexError(f"Row index {key} out of range")
        elif isinstance(key, str):
            for row in self:
                if row.id == key:
                    return row
            raise DatasetIndexError(f"Row id '{key}' not found")
        raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self):
        return len(self.rows)


def subset():
    """Marks a Dataset method as a subset filter"""

    def decorator(func):
        setattr(func, SUBSET_FILTER_FN_INDICATOR, True)
        return func

    return decorator
