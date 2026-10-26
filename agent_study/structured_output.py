from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskSchema(BaseModel):
    """给 LLM 结构化输出用的任务契约。

    长度上限按**原始字符串**计（Pydantic 的约束先于 ``field_validator`` 生效），
    这样校验口径和 JSON Schema 里的 ``maxLength`` 完全一致——提示词告诉模型
    "最多 100 个字符"，校验就用同一把尺子量。strip 只会让长度变短，不会越界。

    多余字段一律判错（``extra="forbid"``），不要的字段不会出现在 JSON Schema 里。
    """

    # 多出来的字段直接判错，而不是悄悄丢掉：模型自作主张加一个 due_date，
    # 调用方却以为"这个字段契约里有"，才是最坏的情况——数据看着进去了，
    # 读的时候才发现字段不存在。顺带让 JSON Schema 带上
    # additionalProperties: false，提示词里也跟着收紧。
    model_config = ConfigDict(extra="forbid")

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
