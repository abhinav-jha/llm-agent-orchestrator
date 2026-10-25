from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agentdb"
    redis_url: str = "redis://localhost:6379/0"
    gemini_api_key: str = ""
    tavily_api_key: str = ""
    jwt_private_key_path: str = ".secrets/jwt-private.pem"
    jwt_public_key_path: str = ".secrets/jwt-public.pem"
    access_token_expire_minutes: int = 30
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_request_timeout_seconds: float = 30.0
    web_search_timeout_seconds: float = 12.0
    agent_max_tool_rounds: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
