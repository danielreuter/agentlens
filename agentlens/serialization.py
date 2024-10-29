from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod

from pydantic import BaseModel
from pydantic_core.core_schema import with_info_plain_validator_function


class Serializable(ABC):
    @abstractmethod
    def model_dump(self) -> dict: ...

    @classmethod
    @abstractmethod
    def model_validate(cls, value) -> Serializable: ...

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        return with_info_plain_validator_function(lambda value, info: cls.model_validate(value))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Serializable):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    def __hash__(self) -> int:
        return hash(json.dumps(self.model_dump(), sort_keys=True))

    def __getstate__(self):
        return self.model_dump()

    def __setstate__(self, state):
        obj = self.model_validate(state)
        self.__dict__.update(obj.__dict__)

    def test_serialized_to_json(self, dumped: dict):
        try:
            json.dumps(dumped)
        except Exception as e:
            raise ValueError(f"model_dump() returned a value that is not serializable to JSON: {e}")

    def test_serialization_roundtrip(self, dumped: dict):
        try:
            reconstructed = self.model_validate(dumped)
            redumped = reconstructed.model_dump()
            assert dumped == redumped, (
                "Serialization roundtrip failed! "
                "Check that `model_dump()` and `model_validate()` are inverses of each other."
            )
        except Exception as e:
            raise ValueError(f"Serialization roundtrip failed! {e}")

    def test_pickle_roundtrip(self, dumped: dict):
        try:
            pickled = pickle.dumps(dumped)
            unpickled = pickle.loads(pickled)
            assert dumped == unpickled, (
                "Pickle roundtrip failed! " "The unpickled object is not equal to the original."
            )
        except Exception as e:
            raise ValueError(f"Pickle serialization error: {e}")

    def test_serialization(self):
        dumped = self.model_dump()
        self.test_serialized_to_json(dumped)
        self.test_serialization_roundtrip(dumped)
        self.test_pickle_roundtrip(dumped)


SerializableType = BaseModel | Serializable | str | int | float | bool | dict | list


class TaskInput(BaseModel):
    args: tuple[SerializableType, ...]
    kwargs: dict[str, SerializableType]


class TaskOutput(BaseModel):
    return_value: SerializableType


def serialize_task_input(*args, **kwargs) -> dict:
    return TaskInput(args=args, kwargs=kwargs).model_dump()


def serialize_task_output(return_value) -> dict:
    return TaskOutput(return_value=return_value).model_dump()
