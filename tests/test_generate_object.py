from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Think(BaseModel, Generic[T]):
    reasoning: str = Field(description="Your reasoning")
    action: T = Field(description="The next action to be taken")


# @pytest.mark.asyncio
# async def test_generate_object(ai: AI):
#     response = await ai.generate_object(model="openai:gpt-4o", type=Think[str], prompt="sup")
