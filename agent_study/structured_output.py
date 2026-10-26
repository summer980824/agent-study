from typing import Literal

from pydantic import BaseModel, Field, field_validator


class TaskSchema(BaseModel):
    """给 LLM 结构化输出用的任务契约。

    长度上限按**原始字符串**计（Pydantic 的约束先于 ``field_validator`` 生效），
    这样校验口径和 JSON Schema 里的 ``maxLength`` 完全一致——提示词告诉模型
    "最多 100 个字符"，校验就用同一把尺子量。strip 只会让长度变短，不会越界。
    """

    title: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1)
    priority: Literal["low", "medium", "high"]
    tags: list[str] = Field(default_factory=list)
    estimated_minutes: int = Field(gt=0, le=480)

    @field_validator("title", "description")
    @classmethod
    def not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("value cannot be blank")
        return stripped
