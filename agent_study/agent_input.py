from pydantic import BaseModel, Field


class AgentInput(BaseModel):
    user_message: str = Field(min_length=1)
    context: list[str] = []
    max_steps: int = Field(ge=1, le=50, default=10)