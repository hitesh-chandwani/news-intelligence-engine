"""Tests for src/nie/config.py.

No network, DB, or non-fixture filesystem access: settings are loaded only
from fixture `.env` files under tests/fixtures/config/, per
_docs/testing-guidelines.md.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from nie.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "config"

# Names of all env vars Settings reads, so tests can isolate from whatever
# happens to be set in the real process environment.
_ALL_ENV_VARS = [
    "DATABASE_URL",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "RESEND_API_KEY",
    "NOTIFY_EMAIL_TO",
    "DISCOVERY_PROVIDERS",
    "RSS_FEEDS",
    "POLL_INTERVAL_MINUTES",
    "NOTIFY_MIN_IMPORTANCE",
    "DEDUP_WINDOW_DAYS",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the real shell environment never leaks into these tests."""
    for name in _ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_settings_load_all_values_from_full_fixture() -> None:
    settings = Settings(_env_file=str(FIXTURES_DIR / "full.env"))  # type: ignore[call-arg]

    assert settings.database_url == "postgresql+asyncpg://testuser:testpass@testhost:5432/testdb"
    assert settings.llm_base_url == "https://example-llm.test/v1"
    assert settings.llm_api_key == "test-llm-key"
    assert settings.llm_model == "test-model"
    assert settings.embedding_model == "test-embedding-model"
    assert settings.telegram_bot_token == "test-bot-token"
    assert settings.telegram_chat_id == "test-chat-id"
    assert settings.resend_api_key == "test-resend-key"
    assert settings.notify_email_to == "alerts@example.test"
    assert settings.discovery_providers == ["stub", "rss", "extra"]
    assert settings.rss_feeds == ["https://feed.one/rss", "https://feed.two/rss"]
    assert settings.poll_interval_minutes == 15
    assert isinstance(settings.poll_interval_minutes, int)
    assert settings.notify_min_importance == "high"
    assert settings.dedup_window_days == 7
    assert isinstance(settings.dedup_window_days, int)


def test_settings_load_with_only_defaulted_keys_set() -> None:
    settings = Settings(_env_file=str(FIXTURES_DIR / "defaults_only.env"))  # type: ignore[call-arg]

    # The 8 defaulted values load from the fixture as expected.
    assert (
        settings.database_url
        == "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/nie_db"
    )
    assert settings.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert settings.llm_model == "gemini-2.5-flash"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.discovery_providers == ["stub", "rss"]
    assert settings.poll_interval_minutes == 60
    assert settings.notify_min_importance == "medium"
    assert settings.dedup_window_days == 14

    # The 6 variables absent from the fixture (no design.md default) do not
    # block loading, and come back None/empty.
    assert settings.llm_api_key is None
    assert settings.telegram_bot_token is None
    assert settings.telegram_chat_id is None
    assert settings.resend_api_key is None
    assert settings.notify_email_to is None
    assert settings.rss_feeds == []


def test_invalid_poll_interval_minutes_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLL_INTERVAL_MINUTES", "not-a-number")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)  # type: ignore[call-arg]

    assert "poll_interval_minutes" in str(exc_info.value)


def test_invalid_dedup_window_days_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEDUP_WINDOW_DAYS", "not-a-number")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)  # type: ignore[call-arg]

    assert "dedup_window_days" in str(exc_info.value)


def test_invalid_notify_min_importance_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTIFY_MIN_IMPORTANCE", "urgent")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)  # type: ignore[call-arg]

    assert "notify_min_importance" in str(exc_info.value)
