"""Tests for `notify_stage` (#48, `src/nie/pipeline/notify.py`).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_notify.py`'s/
`tests/test_pipeline_relate.py`'s `migrated_db`/`session_factory` fixture
pattern -- a fresh engine per test, not the module-level singleton, so
pooled asyncpg connections stay bound to this test's own event loop.

`notify_stage`'s selection query is global -- no `watch_id` filter, by
design (#48's own selection-query spec). The Compose Postgres is shared
and never truncated between test runs, so a stray `event` row left
selectable by an earlier test run (this file's own, or
`tests/test_pipeline_notify.py`'s/`tests/test_pipeline_score.py`'s/
`tests/test_pipeline_relate.py`'s) would otherwise leak into a later
test's selection -- same global-selection-query test-pollution precedent
`tests/test_pipeline_relate.py`'s `_clear_stale_scored_unrelated_events`
sets, here via `_clear_stale_notify_candidates` (see its own docstring
for the mechanism).

`send_email`/`send_telegram_message` are monkeypatched at their defining
modules (`nie.notifications.email.send_email` /
`nie.notifications.telegram.send_telegram_message`) rather than on
`nie.pipeline.notify`, since `notify_stage` imports them locally, inside
its own function body, on every call -- same stubbed-network-call
precedent `tests/test_notifications_email.py`/
`tests/test_notifications_telegram.py` set for these two functions, no
live Resend/Telegram calls.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Event, Notification, NotificationPreference, Watch
from nie.notifications.email import EmailSendError
from nie.pipeline.notify import build_notification_payload, notify_stage

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


@pytest.fixture(autouse=True)
async def _clear_stale_notify_candidates(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Move every pre-existing `notify_stage`-selectable `event` row out of
    selection, by inserting a fresh dummy `Notification` row (`created_at`
    defaulting to "now") for each.

    A dummy row's `created_at` is always `>=` any pre-existing
    `Event.last_material_update_at` -- nothing in this test suite ever
    back-dates that column into the future -- so inserting one neutralizes
    both selection conditions at once for that event: it now has a
    `Notification` row (no longer a "new-event" candidate), and that
    row's `created_at` is never earlier than `last_material_update_at`
    (no longer a "material-update" candidate either). Runs before each
    test's own event rows are created, so it only ever touches
    pre-existing rows, never the test's own fixtures -- same
    global-selection-query test-pollution precedent
    `tests/test_pipeline_relate.py`'s `_clear_stale_scored_unrelated_events`
    sets, adapted to `notify_stage`'s own selection query rather than
    reusing that one.
    """
    async with session_factory() as session:
        last_notification = (
            select(
                Notification.event_id.label("event_id"),
                func.max(Notification.created_at).label("last_notified_at"),
            )
            .group_by(Notification.event_id)
            .subquery()
        )
        result = await session.execute(
            select(Event.id, Event.watch_id)
            .outerjoin(last_notification, last_notification.c.event_id == Event.id)
            .where(
                or_(
                    last_notification.c.last_notified_at.is_(None),
                    Event.last_material_update_at > last_notification.c.last_notified_at,
                )
            )
        )
        stale = result.all()
        for event_id, watch_id in stale:
            session.add(
                Notification(
                    watch_id=watch_id,
                    event_id=event_id,
                    reason="new-event",
                    payload={},
                    channels_sent=[],
                )
            )
        await session.commit()


async def _make_watch(session: AsyncSession, prefix: str = "notify-stage-test") -> Watch:
    watch = Watch(slug=unique_slug(prefix), name="Notify Stage Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch


def _make_scored_event(
    watch_id: uuid.UUID,
    *,
    title: str = "Some Event",
    relevance: str | None = "high",
    importance: str | None = "critical",
) -> Event:
    return Event(
        watch_id=watch_id,
        title=title,
        fact_summary="A fact summary.",
        interpretation="An interpretation.",
        entities=[],
        embedding=ZERO_EMBEDDING,
        relevance=relevance,
        importance=importance,
        impact_direction="bullish",
        impact_reason="Because of the rationale.",
        impact_confidence="high",
    )


def _make_preference(
    watch_id: uuid.UUID,
    *,
    min_importance: str = "medium",
    categories: list[str] | None = None,
    channels: list[str] | None = None,
) -> NotificationPreference:
    return NotificationPreference(
        watch_id=watch_id,
        min_importance=min_importance,
        categories=categories if categories is not None else [],
        channels=channels if channels is not None else [],
    )


async def _notifications_for(session: AsyncSession, event_id: uuid.UUID) -> list[Notification]:
    result = await session.execute(select(Notification).where(Notification.event_id == event_id))
    return list(result.scalars())


# ---------------------------------------------------------------------------
# new-event candidate
# ---------------------------------------------------------------------------


async def test_new_event_candidate_passes_gate_produces_one_notification(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        session.add(_make_preference(watch.id, channels=["inapp"]))
        event = _make_scored_event(watch.id)
        session.add(event)
        await session.commit()

        expected_payload = await build_notification_payload(session, event)

        result = await notify_stage(session)
        await session.commit()

        rows = await _notifications_for(session, event.id)

    assert result["notified"] == 1
    assert result["gated_out"] == 0
    assert result["skipped_no_preference"] == 0

    assert len(rows) == 1
    row = rows[0]
    assert row.watch_id == watch.id
    assert row.reason == "new-event"
    assert row.channels_sent == ["inapp"]
    assert row.payload == expected_payload.model_dump(mode="json")


# ---------------------------------------------------------------------------
# material-update candidate
# ---------------------------------------------------------------------------


async def test_material_update_candidate_is_re_notified(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime.now(UTC)

    async with session_factory() as session:
        watch = await _make_watch(session)
        session.add(_make_preference(watch.id, channels=["inapp"]))
        event = _make_scored_event(watch.id)
        session.add(event)
        await session.commit()

        # An existing notification, dated well in the past.
        session.add(
            Notification(
                watch_id=watch.id,
                event_id=event.id,
                reason="new-event",
                payload={},
                channels_sent=[],
                created_at=now - timedelta(days=2),
            )
        )
        await session.commit()

        # A material update after that notification.
        event.last_material_update_at = now - timedelta(days=1)
        await session.commit()

        result = await notify_stage(session)
        await session.commit()

        rows = await _notifications_for(session, event.id)

    assert result["notified"] == 1
    assert len(rows) == 2
    new_row = max(rows, key=lambda row: row.created_at)
    assert new_row.reason == "material-update"
    assert new_row.channels_sent == ["inapp"]


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


async def test_already_notified_at_current_state_not_reselected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        session.add(_make_preference(watch.id, channels=["inapp"]))
        event = _make_scored_event(watch.id)
        session.add(event)
        await session.commit()

        first_result = await notify_stage(session)
        await session.commit()

        # No change to `event.last_material_update_at` in between.
        second_result = await notify_stage(session)
        await session.commit()

        rows = await _notifications_for(session, event.id)

    assert first_result["notified"] == 1
    assert second_result["notified"] == 0
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# missing preference
# ---------------------------------------------------------------------------


async def test_event_whose_watch_has_no_preference_is_skipped(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        # Deliberately no `NotificationPreference` row for this watch.
        event = _make_scored_event(watch.id)
        session.add(event)
        await session.commit()

        result = await notify_stage(session)
        await session.commit()

        rows = await _notifications_for(session, event.id)

    assert result["notified"] == 0
    assert result["skipped_no_preference"] == 1
    assert rows == []


# ---------------------------------------------------------------------------
# gate returns False
# ---------------------------------------------------------------------------


async def test_gate_false_produces_no_notification(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        session.add(_make_preference(watch.id, channels=["inapp"]))
        event = _make_scored_event(watch.id, relevance="irrelevant")
        session.add(event)
        await session.commit()

        result = await notify_stage(session)
        await session.commit()

        rows = await _notifications_for(session, event.id)

    assert result["notified"] == 0
    assert result["gated_out"] == 1
    assert rows == []


# ---------------------------------------------------------------------------
# per-channel failure isolation
# ---------------------------------------------------------------------------


async def test_one_channel_failing_still_inserts_row_with_surviving_channels(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telegram_calls = []

    async def fake_send_email(payload: object, settings: object = None) -> None:
        raise EmailSendError("simulated Resend failure")

    async def fake_send_telegram_message(payload: object, settings: object = None) -> None:
        telegram_calls.append(payload)

    monkeypatch.setattr("nie.notifications.email.send_email", fake_send_email)
    monkeypatch.setattr(
        "nie.notifications.telegram.send_telegram_message", fake_send_telegram_message
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        session.add(_make_preference(watch.id, channels=["email", "telegram"]))
        event = _make_scored_event(watch.id)
        session.add(event)
        await session.commit()

        result = await notify_stage(session)
        await session.commit()

        rows = await _notifications_for(session, event.id)

    assert result["notified"] == 1
    assert len(telegram_calls) == 1
    assert len(rows) == 1
    assert rows[0].channels_sent == ["telegram"]
