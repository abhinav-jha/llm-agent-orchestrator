from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agentdb"
    redis_url: str = "redis://localhost:6379/0"
    gemini_api_key: str = ""
    jwt_private_key_path: str = ".secrets/jwt-private.pem"
    jwt_public_key_path: str = ".secrets/jwt-public.pem"
    access_token_expire_minutes: int = 60
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    agent_max_tool_rounds: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
