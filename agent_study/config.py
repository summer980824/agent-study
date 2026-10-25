import os
from pathlib import Path

from pydantic import ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（agent_study 的上级目录），保证从任何工作目录启动都能找到 .env
BASE_DIR = Path(__file__).resolve().parent.parent

ENV = os.getenv("APP_ENV", "dev")

# 官方默认地址（见 MiniMax 开放平台文档）：
# - OpenAI 兼容接口：https://api.minimax.io/v1
# - Anthropic 兼容接口：https://api.minimax.io/anthropic
DEFAULT_MODEL_NAME = "MiniMax-M3.1-Flash"
DEFAULT_API_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.minimax.io/anthropic"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[BASE_DIR / ".env", BASE_DIR / f".env.{ENV}"],  # 先读公共 .env，.env.dev/.env.prod 覆盖
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_NAME: str = DEFAULT_MODEL_NAME
    API_KEY: str
    # OpenAI 兼容接口地址，llm_client 用
    API_BASE_URL: str = DEFAULT_API_BASE_URL
    # Anthropic 兼容接口地址，anthropic_chat 用；与上面是两种协议，不能互相顶替
    ANTHROPIC_BASE_URL: str = DEFAULT_ANTHROPIC_BASE_URL

    @field_validator("MODEL_NAME", "API_BASE_URL", "ANTHROPIC_BASE_URL", mode="before")
    @classmethod
    def _blank_to_default(cls, value: object, info: ValidationInfo) -> object:
        """空值（``.env`` 里写成 ``KEY=``）一律回退到默认值。"""
        if value is None or (isinstance(value, str) and not value.strip()):
            return cls.model_fields[info.field_name].default
        return value

    @field_validator("API_KEY")
    @classmethod
    def _key_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("API_KEY 不能为空")
        return value


settings = Settings()
