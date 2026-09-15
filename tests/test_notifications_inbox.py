"""Tests for src/nie/notifications/inbox.py (issue #33).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked -- a `Watch`, `Event`, and several
`Notification` rows are real inserted rows. Follows
`tests/test_pipeline_notify.py`'s/`tests/test_pipeline_relate.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test, not the module-level singleton, so pooled asyncpg connections stay
bound to this test's own event loop.

`created_at` values on `Notification` rows are always explicitly set
(never left to `server_default=func.now()`), per the issue's ordering
test note -- distinct, hand-chosen timestamps avoid same-timestamp
ordering flakiness.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Event, Notification, Watch
from nie.notifications.inbox import list_notifications, mark_read

REPO_ROOT = Path(__file__).parent.parent

EMBEDDING_DIM = 384
ZERO_EMBEDDING = [0.0] * EMBEDDING_DIM


def unique_slug(prefix: str) -> str:
    """A per-run-unique slug so tests stay independent of prior DB state."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


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


async def _make_watch(session: AsyncSession, prefix: str = "inbox-test") -> Watch:
    watch = Watch(slug=unique_slug(prefix), name="Inbox Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch


async def _make_event(session: AsyncSession, watch_id: uuid.UUID, title: str) -> Event:
    event = Event(
        watch_id=watch_id,
        title=title,
        fact_summary="A fact summary.",
        interpretation="An interpretation.",
        entities=[],
        embedding=ZERO_EMBEDDING,
    )
    session.add(event)
    await session.commit()
    return event


def _make_notification(
    *,
    watch_id: uuid.UUID,
    event_id: uuid.UUID,
    created_at: datetime,
) -> Notification:
    return Notification(
        watch_id=watch_id,
        event_id=event_id,
        reason="new-event",
        payload={"title": "Something happened"},
        channels_sent=[],
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# list_notifications
# ---------------------------------------------------------------------------


async def test_list_notifications_returns_newest_first(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        event = await _make_event(session, watch.id, "Some Event")

        base = datetime(2026, 1, 1, tzinfo=UTC)
        oldest = _make_notification(
            watch_id=watch.id, event_id=event.id, created_at=base
        )
        newest = _make_notification(
            watch_id=watch.id, event_id=event.id, created_at=base + timedelta(hours=2)
        )
        middle = _make_notification(
            watch_id=watch.id, event_id=event.id, created_at=base + timedelta(hours=1)
        )
        # Inserted out of chronological order on purpose.
        session.add_all([oldest, newest, middle])
        await session.commit()

        result = await list_notifications(session, watch.id)

        assert [n.id for n in result] == [newest.id, middle.id, oldest.id]


async def test_list_notifications_empty_for_watch_with_no_notifications(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        empty_watch = await _make_watch(session, prefix="inbox-test-empty")

        result = await list_notifications(session, empty_watch.id)

        assert result == []


async def test_list_notifications_empty_for_unknown_watch_id(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        result = await list_notifications(session, uuid.uuid4())

        assert result == []


# ---------------------------------------------------------------------------
# mark_read
# ---------------------------------------------------------------------------


async def test_mark_read_sets_read_at_and_is_durable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        event = await _make_event(session, watch.id, "Some Event")
        notification = _make_notification(
            watch_id=watch.id, event_id=event.id, created_at=datetime.now(UTC)
        )
        session.add(notification)
        await session.commit()

        assert notification.read_at is None

        updated = await mark_read(session, notification.id)

        assert updated.read_at is not None
        assert updated.read_at.tzinfo is not None
        first_read_at = updated.read_at

    # Durability: re-fetch in a brand-new session/engine.
    async with session_factory() as fresh_session:
        refetched = await fresh_session.get(Notification, notification.id)
        assert refetched is not None
        assert refetched.read_at == first_read_at


async def test_mark_read_second_call_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        event = await _make_event(session, watch.id, "Some Event")
        notification = _make_notification(
            watch_id=watch.id, event_id=event.id, created_at=datetime.now(UTC)
        )
        session.add(notification)
        await session.commit()

        first = await mark_read(session, notification.id)
        first_read_at = first.read_at
        assert first_read_at is not None

        second = await mark_read(session, notification.id)

        assert second.read_at == first_read_at


async def test_mark_read_raises_value_error_for_unknown_id(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        with pytest.raises(ValueError):
            await mark_read(session, uuid.uuid4())
