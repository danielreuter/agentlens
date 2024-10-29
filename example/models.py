from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image
from pydantic import BaseModel

from agentlens.serialization import Serializable


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
