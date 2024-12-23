from pydantic import BaseModel


class Counter(BaseModel):
    value: int = 0


class Messages(BaseModel):
    items: list[str] = []
