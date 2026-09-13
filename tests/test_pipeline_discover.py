"""Tests for src/nie/pipeline/discover.py (issue #19).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_runner.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test, not the module-level singleton, so pooled asyncpg connections stay
bound to this test's own event loop.

Env isolated the same way as `tests/test_config.py`'s `_isolate_env`
fixture: `discover_stage` builds its own `Settings()` internally (it
isn't handed one), so the real shell/`.env` must never leak in.
`DISCOVERY_PROVIDERS` is set to `stub` only for the happy-path tests, so
no network call is made and `RssProvider` is never exercised.

Unlike the rest of this suite's `unique_slug`-per-test convention
(`tests/test_models.py`), `SILVER_WATCH_SLUG` is a fixed literal that
`discover_stage` itself looks up, not something a test can vary -- so
this file can't lean on a fresh slug per test for isolation. The Compose
Postgres is also never truncated between test runs (there's no
truncate/drop_all fixture anywhere in this suite), so a previous run of
this same file (e.g. from the team's "run pytest twice in a row"
verification step) can leave the "silver" watch and its `source` rows
behind. Each test below therefore establishes the exact state it needs
itself -- resetting `source` rows before the dedup test, deleting the
watch and everything referencing it before the no-watch test -- instead
of assuming a pristine database.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import ContextItem, NotificationPreference, Source, Watch
from nie.pipeline.discover import discover_stage
from nie.seed.run import SILVER_WATCH_SLUG, seed

REPO_ROOT = Path(__file__).parent.parent

# Names of all env vars Settings reads, so tests can isolate from whatever
# happens to be set in the real process environment -- same list
# `tests/test_config.py` uses.
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

_EXPECTED_URLS = {
    "https://example.com/silver-etf-inflows",
    "https://example.com/mining-output-report",
    "https://example.com/fed-rate-decision-silver",
}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the real shell environment never leaks into these tests."""
    for name in _ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def migrated_db() -> None:
    """Run `alembic upgrade head` against the live Compose DB.

    A plain (sync) fixture: Alembic drives its own event loop internally,
    so this must run outside pytest-asyncio's loop for the test function.
    Re-running this is idempotent -- Alembic is a no-op when the database
    is already at the target revision.
    """
    config = Config(str(REPO_ROOT / "alembic.ini"))
    upgrade(config, "head")


@pytest.fixture
async def session_factory(migrated_db: None) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory built from its own engine, scoped to this test."""
    engine = create_engine()
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


async def _get_silver_watch_id(session: AsyncSession) -> uuid.UUID | None:
    result = await session.execute(select(Watch.id).where(Watch.slug == SILVER_WATCH_SLUG))
    return result.scalar_one_or_none()


async def _delete_silver_watch_and_dependents(session: AsyncSession) -> None:
    """Delete the "silver" `Watch` row and everything referencing it.

    Not part of `discover_stage`'s own behaviour -- this suite's DB is
    never truncated between test runs (see module docstring), so a test
    that needs "no watch exists" has to establish that state itself
    rather than assume it.
    """
    watch_id = await _get_silver_watch_id(session)
    if watch_id is None:
        return
    await session.execute(delete(Source).where(Source.watch_id == watch_id))
    await session.execute(delete(ContextItem).where(ContextItem.watch_id == watch_id))
    await session.execute(
        delete(NotificationPreference).where(NotificationPreference.watch_id == watch_id)
    )
    await session.execute(delete(Watch).where(Watch.id == watch_id))
    await session.commit()


async def test_discover_stage_inserts_stub_fixtures_and_dedupes_on_rerun(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCOVERY_PROVIDERS", "stub")

    async with session_factory() as session:
        await seed(session)
        watch_id = await _get_silver_watch_id(session)
        assert watch_id is not None
        # Reset to a clean slate: a previous run of this same test (e.g.
        # from re-running the suite) leaves these 3 rows behind, since
        # discover_stage never deletes sources.
        await session.execute(delete(Source).where(Source.watch_id == watch_id))
        await session.commit()

    async with session_factory() as session:
        result = await discover_stage(session)
        await session.commit()
    assert result == {"discovered": 3}

    async with session_factory() as session:
        sources_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        sources = sources_result.scalars().all()
        assert len(sources) == 3
        assert {source.url for source in sources} == _EXPECTED_URLS
        assert {source.status for source in sources} == {"discovered"}
        assert {source.watch_id for source in sources} == {watch_id}

        by_url = {source.url: source for source in sources}
        # One fixture sets "entities" explicitly; the other two omit it
        # and must come back as [] (never None -- the column is NOT NULL).
        assert by_url["https://example.com/silver-etf-inflows"].entities == [
            "Silver",
            "ETF",
            "Kitco",
        ]
        assert by_url["https://example.com/mining-output-report"].entities == []
        assert by_url["https://example.com/fed-rate-decision-silver"].entities == []
        # A fixture that already has content still lands as "discovered",
        # not some other status -- populating content/moving status off
        # "discovered" is #20's job, not this stage's.
        assert (
            by_url["https://example.com/fed-rate-decision-silver"].content is not None
        )
        assert by_url["https://example.com/mining-output-report"].content is None

    async with session_factory() as session:
        result_again = await discover_stage(session)
        await session.commit()
    assert result_again == {"discovered": 0}

    async with session_factory() as session:
        sources_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        assert len(sources_result.scalars().all()) == 3


async def test_discover_stage_raises_when_no_watch_exists(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCOVERY_PROVIDERS", "stub")

    async with session_factory() as session:
        await _delete_silver_watch_and_dependents(session)

        with pytest.raises(LookupError):
            await discover_stage(session)


async def test_discover_stage_raises_for_unrecognized_provider_name(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISCOVERY_PROVIDERS", "stub,not-a-real-provider")

    async with session_factory() as session:
        with pytest.raises(ValueError, match="not-a-real-provider"):
            await discover_stage(session)
