from pydantic import BaseModel


class ModelUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class InferenceCost(BaseModel):
    input_cost: float = 0.0
    output_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost
