"""Tests for src/nie/pipeline/match.py (issue #22).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_embed.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test, not the module-level singleton, so pooled asyncpg connections stay
bound to this test's own event loop.

`find_candidate_events` builds its own `Settings()` internally (it isn't
handed one, same pattern `discover_stage` uses), so `_isolate_env`
(copied from `tests/test_pipeline_discover.py`) keeps the real shell/
`.env` from leaking `DEDUP_WINDOW_DAYS` (or anything else) into these
tests -- the default `dedup_window_days=14` is what every "inside"/
"outside the window" fixture below is built against. Each test creates
its own `Watch` row(s) with `unique_slug` (same helper `tests/
test_models.py` defines), keeping tests independent of each other and of
any prior run's leftover rows (the Compose Postgres is never truncated
between test runs).

No test here makes a live network call or a real embedding call: every
"near"/"far" test vector is a hand-authored fixed 384-length float list
(mostly zeros with one or two components set), not a FastEmbed output --
this file tests the SQL/filtering/ordering logic, not embedding quality.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Event, Source, Watch
from nie.pipeline.match import MatchCandidate, find_candidate_events

REPO_ROOT = Path(__file__).parent.parent

# Same list `tests/test_pipeline_discover.py` uses, so the real shell/
# `.env` never leaks into `Settings()` calls made inside these tests.
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

EMBEDDING_DIM = 384
# Default `Settings().dedup_window_days` (14) with `_isolate_env` in play.
DEDUP_WINDOW_DAYS = 14


def unique_slug(prefix: str) -> str:
    """A per-run-unique slug so tests stay independent of prior DB state.

    Same helper `tests/test_models.py`/`tests/test_pipeline_embed.py`
    define -- there's no row-cleanup fixture, so each test generates its
    own `Watch.slug` rather than relying on a fixed one.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _vector(**components: float) -> list[float]:
    """A fixed 384-length float list with the given indices set and every
    other component zero -- a hand-authored stand-in for a real
    embedding, never a live FastEmbed call.
    """
    vec = [0.0] * EMBEDDING_DIM
    for index, value in components.items():
        vec[int(index)] = value
    return vec


# The source's embedding: a unit vector along axis 0.
SOURCE_EMBEDDING = _vector(**{"0": 1.0})
# Identical direction to the source -- cosine distance 0.
NEAR_EMBEDDING_DIST_0 = _vector(**{"0": 1.0})
# 45 degrees off the source -- cosine distance 1 - 1/sqrt(2) ~= 0.29289.
NEAR_EMBEDDING_DIST_1 = _vector(**{"0": 1.0, "1": 1.0})
# Further off the source -- cosine distance 1 - 1/sqrt(5) ~= 0.55279.
NEAR_EMBEDDING_DIST_2 = _vector(**{"0": 1.0, "1": 2.0})


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


async def _make_watch(session: AsyncSession, prefix: str = "match-test") -> uuid.UUID:
    watch = Watch(slug=unique_slug(prefix), name="Match Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch.id


def _make_source(
    watch_id: uuid.UUID, *, embedding: list[float] | None
) -> Source:
    return Source(
        watch_id=watch_id,
        url=f"https://example.com/{uuid.uuid4().hex[:8]}",
        title="Some Source",
        source_name="Example",
        entities=[],
        status="extracted" if embedding is None else "processed",
        embedding=embedding,
    )


def _make_event(
    watch_id: uuid.UUID,
    *,
    embedding: list[float],
    event_date: datetime | None,
    title: str = "Some Event",
) -> Event:
    return Event(
        watch_id=watch_id,
        title=title,
        fact_summary="Some fact summary.",
        interpretation="Some interpretation.",
        event_date=event_date,
        entities=[],
        embedding=embedding,
    )


async def test_find_candidate_events_scopes_by_watch_and_coalesced_window(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime.now(UTC)
    inside_window = now - timedelta(days=1)
    outside_window = now - timedelta(days=DEDUP_WINDOW_DAYS + 5)

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        other_watch_id = await _make_watch(session, prefix="match-test-other")

        source = _make_source(watch_id, embedding=SOURCE_EMBEDDING)
        session.add(source)

        # 1. Near, event_date inside the window -- included, nearest.
        event_included_by_date = _make_event(
            watch_id,
            embedding=NEAR_EMBEDDING_DIST_0,
            event_date=inside_window,
            title="Included via event_date",
        )
        # 2. Near, event_date NULL, discovered_at (server default "now")
        #    inside the window -- included via the COALESCE fallback.
        event_included_by_coalesce = _make_event(
            watch_id,
            embedding=NEAR_EMBEDDING_DIST_1,
            event_date=None,
            title="Included via discovered_at fallback",
        )
        # 3. Near, event_date outside the window -- excluded.
        event_excluded_by_date = _make_event(
            watch_id,
            embedding=NEAR_EMBEDDING_DIST_0,
            event_date=outside_window,
            title="Excluded via old event_date",
        )
        # 4. Near, event_date inside the window, but a different watch --
        #    excluded by watch scoping.
        event_excluded_by_watch = _make_event(
            other_watch_id,
            embedding=NEAR_EMBEDDING_DIST_0,
            event_date=inside_window,
            title="Excluded via different watch",
        )
        session.add_all(
            [
                event_included_by_date,
                event_included_by_coalesce,
                event_excluded_by_date,
                event_excluded_by_watch,
            ]
        )
        await session.commit()
        await session.refresh(source)

        candidates = await find_candidate_events(session, source)

    assert all(isinstance(candidate, MatchCandidate) for candidate in candidates)
    assert [candidate.event_id for candidate in candidates] == [
        event_included_by_date.id,
        event_included_by_coalesce.id,
    ]
    assert candidates[0].distance == pytest.approx(0.0, abs=1e-4)
    assert candidates[1].distance == pytest.approx(1 - 1 / (2**0.5), abs=1e-4)
    assert candidates[0].distance < candidates[1].distance
    returned_ids = {candidate.event_id for candidate in candidates}
    assert event_excluded_by_date.id not in returned_ids
    assert event_excluded_by_watch.id not in returned_ids


async def test_find_candidate_events_truncates_to_limit(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime.now(UTC)
    inside_window = now - timedelta(days=1)

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(watch_id, embedding=SOURCE_EMBEDDING)
        session.add(source)

        nearest = _make_event(
            watch_id, embedding=NEAR_EMBEDDING_DIST_0, event_date=inside_window, title="Nearest"
        )
        middle = _make_event(
            watch_id, embedding=NEAR_EMBEDDING_DIST_1, event_date=inside_window, title="Middle"
        )
        farthest = _make_event(
            watch_id, embedding=NEAR_EMBEDDING_DIST_2, event_date=inside_window, title="Farthest"
        )
        session.add_all([nearest, middle, farthest])
        await session.commit()
        await session.refresh(source)

        candidates = await find_candidate_events(session, source, limit=2)

    assert len(candidates) == 2
    assert [candidate.event_id for candidate in candidates] == [nearest.id, middle.id]
    assert candidates[0].distance < candidates[1].distance


async def test_find_candidate_events_raises_value_error_on_null_embedding_without_querying(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(watch_id, embedding=None)
        session.add(source)
        await session.commit()
        await session.refresh(source)

        execute_call_count = 0
        original_execute = session.execute

        async def counting_execute(*args: Any, **kwargs: Any) -> Any:
            nonlocal execute_call_count
            execute_call_count += 1
            return await original_execute(*args, **kwargs)

        monkeypatch.setattr(session, "execute", counting_execute)

        with pytest.raises(ValueError, match="embedding"):
            await find_candidate_events(session, source)

    assert execute_call_count == 0


async def test_find_candidate_events_returns_empty_list_when_nothing_qualifies(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(watch_id, embedding=SOURCE_EMBEDDING)
        session.add(source)
        await session.commit()
        await session.refresh(source)

        # No `event` rows exist for this freshly created watch at all.
        candidates = await find_candidate_events(session, source)

    assert candidates == []
