from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI-Powered Multi-Platform Prospect Discovery API"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 4000
    log_level: str = "INFO"

    # LLM configuration
    llm_provider: Literal["gemini", "openai", "anthropic", "auto"] = "gemini"
    llm_model: str = "gemini-2.5-flash-lite"
    llm_fallback_provider: Optional[Literal["gemini", "openai", "anthropic"]] = "openai"
    llm_fallback_model: str = "gpt-4.1-nano"
    llm_timeout_seconds: float = 20.0
    llm_temperature: float = 0.1

    gemini_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)

    # Search & Platform API credentials
    serper_api_key: Optional[str] = Field(default=None)
    brave_search_api_key: Optional[str] = Field(default=None)
    tavily_api_key: Optional[str] = Field(default=None)
    youtube_api_key: Optional[str] = Field(default=None)
    google_cse_api_key: Optional[str] = Field(default=None)
    google_cse_cx: Optional[str] = Field(default=None)

    # Discovery & performance controls
    max_results_per_platform: int = Field(default=6, ge=1, le=20)
    max_prospects_returned: int = Field(default=20, ge=1, le=100)
    http_timeout_seconds: float = Field(default=12.0, ge=2.0, le=60.0)
    max_concurrent_requests: int = Field(default=8, ge=1, le=32)
    enable_page_enrichment: bool = True
    verify_profile_urls: bool = False
    cache_ttl_seconds: int = Field(default=900, ge=0)
    cache_max_size: int = Field(default=256, ge=16)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
