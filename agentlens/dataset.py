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
) -> Any:
    return Field(default=default, **kwargs)


class Example(BaseModel):
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


E = TypeVar("E", bound=Example)


class DatasetMetadata(BaseModel):
    version: int
    created_at: datetime
    last_modified: datetime


class DatasetFile(BaseModel, Generic[E]):
    metadata: DatasetMetadata
    examples: list[E]


class Dataset(Generic[E]):
    name: ClassVar[str]
    dataset_dir: ClassVar[Path]
    subset: str | None
    examples: list[E]
    example: ClassVar[Type[E]]
    _file: DatasetFile[E]

    def __init__(self, subset: str | None = None):
        self.subset = subset
        self._file = self._read_file()
        rows = self._file.examples
        if subset:
            subset_rows, _ = self.split_examples(subset, rows)
            self.examples = subset_rows
        else:
            self.examples = rows

    def run(self):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for base in cls.__orig_bases__:
            if getattr(base, "__origin__", None) is Dataset:
                args = base.__args__
                if args:
                    cls.example = args[0]
                    break
        if cls.example is None:
            raise ValueError("Dataset must specify a row type")

    @property
    def file_path(self) -> Path:
        # stub = f".{self.subset}" if self.subset else ""
        stub = ""
        path = self.dataset_dir / f"{self.name}{stub}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _read_file(self) -> DatasetFile[E]:
        if not self.file_path.exists():
            return DatasetFile(
                metadata=DatasetMetadata(
                    version=1,
                    created_at=datetime.now(),
                    last_modified=datetime.now(),
                ),
                examples=[],
            )
        json_str = self.file_path.read_text()
        return DatasetFile[self.example].model_validate_json(json_str)

    def split_examples(self, subset: str, examples: list[E]) -> tuple[list[E], list[E]]:
        if not hasattr(self, subset):
            raise ValueError(f"Subset filter '{subset}' not found")

        filter_method = getattr(self, subset)
        if not hasattr(filter_method, SUBSET_FILTER_FN_INDICATOR):
            raise ValueError(f"'{subset}' is not a subset filter")

        subset_examples = []
        other_examples = []

        for example in examples:
            if filter_method(example):
                subset_examples.append(example)
            else:
                other_examples.append(example)

        return subset_examples, other_examples

    def extend(self, examples: list[E]) -> None:
        """Appends examples to the dataset in-place"""
        self.examples.extend(examples)

    def clear(self) -> None:
        """Clears the dataset in-place"""
        self.examples.clear()

    def save(self) -> None:
        """
        Saves the current version of the subset to disk, leaving other rows unchanged.
        """
        # concatenate current subset's rows with the rest of the dataset's rows
        new_rows = self.examples
        if self.subset:
            _, other_examples = self.split_examples(self.subset, self._file.examples)
            new_rows.extend(other_examples)

        # overwrite current dataset with new rows
        self._file.examples = new_rows
        json_str = self._file.model_dump_json(indent=2, exclude_defaults=True)
        self.file_path.write_text(json_str)

    def __iter__(self) -> Iterator[E]:
        return iter(self.examples)

    def __getitem__(self, key: Union[int, str, Any]) -> E:
        if isinstance(key, int):
            if 0 <= key < len(self.examples):
                return self.examples[key]
            raise DatasetIndexError(f"Row index {key} out of range")
        elif isinstance(key, str):
            for example in self:
                if example.id == key:
                    return example
            raise DatasetIndexError(f"Row id '{key}' not found")
        raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self):
        return len(self.examples)


def subset():
    """Marks a Dataset method as a subset filter"""

    def decorator(func):
        setattr(func, SUBSET_FILTER_FN_INDICATOR, True)
        return func

    return decorator
