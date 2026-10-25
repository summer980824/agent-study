import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（agent_study 的上级目录），保证从任何工作目录启动都能找到 .env
BASE_DIR = Path(__file__).resolve().parent.parent

ENV = os.getenv("APP_ENV", "dev")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[BASE_DIR / ".env", BASE_DIR / f".env.{ENV}"],  # 先读公共 .env，.env.dev/.env.prod 覆盖
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_NAME: str
    API_KEY: str
    API_BASE_URL: str
    # Anthropic 兼容端点（如 https://api.minimax.cn/anthropic），留空则回退到 API_BASE_URL
    ANTHROPIC_BASE_URL: str = ""


settings = Settings()
