"""Centralized, typed application configuration.

Every environment variable the app reads is defined once, here, as a field
on :class:`Settings`. Other modules should read config via a ``Settings``
instance instead of touching ``os.environ`` directly. See
``_docs/design.md`` §14 for the full variable reference.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _parse_csv_list(value: object) -> object:
    """Split a comma-separated env var string into a stripped list.

    Leaves non-string values (e.g. an already-parsed list) untouched so
    this can be used as a "before" validator without breaking construction
    from kwargs. Unset/empty input becomes an empty list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """Application settings, loaded from the environment and/or a `.env` file.

    Loading requires no network, DB, or filesystem access beyond reading the
    `.env` file itself. None of the fields below without a `design.md`
    default are required to load `Settings()` -- the consuming module for
    each is responsible for validating presence when it actually needs the
    value (see `_docs/design.md` §19 build order).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Variables with a design.md-listed default ---
    database_url: str = "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/nie_db"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    discovery_providers: Annotated[list[str], NoDecode] = ["stub", "rss"]
    poll_interval_minutes: int = 60
    notify_min_importance: Literal["low", "medium", "high", "critical"] = "medium"
    dedup_window_days: int = 14
    max_extract_attempts: int = 3
    max_score_attempts: int = 3
    max_relate_attempts: int = 3

    # --- Variables with no design.md-listed default: optional until the
    # consuming module (see the issue's "Out of scope" list) ships ---
    llm_api_key: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    resend_api_key: str | None = None
    notify_email_to: str | None = None
    rss_feeds: Annotated[list[str], NoDecode] = []

    @field_validator("discovery_providers", "rss_feeds", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        return _parse_csv_list(value)
