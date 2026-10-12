import os
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV = os.getenv("APP_ENV", "dev")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f".env.{ENV}",  # 加载 .env.dev / .env.prod
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_NAME: str
    API_KEY: str
    API_BASE_URL: str


settings = Settings()
