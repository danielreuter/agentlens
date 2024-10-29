# %%

import base64
import json
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Self

from PIL import Image
from pydantic import BaseModel
from pydantic_core.core_schema import with_info_plain_validator_function


class Serializable(ABC):
    def __post_init__(self):
        self.test_serialization()

    @abstractmethod
    def model_dump(self) -> dict: ...

    @abstractmethod
    def model_validate(cls, value) -> Self: ...

    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        return with_info_plain_validator_function(lambda value, info: cls.model_validate(value))

    def test_serialization(self):
        dumped = self.model_dump()
        try:
            json.dumps(dumped)
        except Exception as e:
            raise ValueError(f"model_dump() returned a value that is not serializable to JSON: {e}")

        try:
            reconstructed = self.model_validate(dumped)
            redumped = reconstructed.model_dump()
            assert (
                dumped == redumped
            ), "Serialization roundtrip failed! Check that `model_dump()` and `model_validate()` are inverses of each other."

        except Exception as e:
            raise ValueError(f"Serialization error: {e}")


class PDFPage(Serializable):
    def __init__(self, image: Image.Image):
        self.image = image

    @classmethod
    def model_validate(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, dict) and "image" in value:
            image_bytes = base64.b64decode(value["image"])
            image = Image.open(BytesIO(image_bytes))
            return cls(image=image)
        raise ValueError(f"Cannot convert {value} to PDFPage")

    def model_dump(self):
        with BytesIO() as buffer:
            self.image.save(buffer, format="PNG")
            return {"image": base64.b64encode(buffer.getvalue()).decode()}


class PDF(BaseModel):
    """A serializable PDF document."""

    pages: list[PDFPage]


# %%

test_image = Image.new("RGB", (100, 100), color="red")

pdf = PDF(pages=[PDFPage(test_image), PDFPage(test_image)])  # Two identical pages for testing

pdf_page = PDFPage(test_image)

pdf_page.test_serialization()

# %%
