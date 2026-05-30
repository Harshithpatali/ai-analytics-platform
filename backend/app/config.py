"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    Global application settings.
    """

    APP_NAME: str = "AI Analytics Platform"
    APP_VERSION: str = "1.0.0"

    API_PREFIX: str = "/api"

    database_url: str

    GROQ_API_KEY: str

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]

    class Config:
        env_file = ".env"


settings = Settings()