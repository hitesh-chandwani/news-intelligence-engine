"""Tests for POST /events/{id}/feedback and
DELETE /events/{id}/feedback/{feedback_id} in
src/nie/web/routers/events.py (issues #39, #51).

Follows the exact `migrated_db`/`session_factory`/`silver_watch_id`/
`seeded_client` fixture pattern `tests/test_web_events.py` (#37)/
`tests/test_web_notifications.py` (#38) established: a fresh engine per
test (not `nie.db`'s module-level singleton), `create_app()` +
`app.dependency_overrides[get_session]`, `httpx.AsyncClient` +
`ASGITransport` (no real server process), seeded via `nie.seed.run.seed`.

Every test creates its own `Event` (+ a `Notification` where needed)
directly against the real Silver watch via `session.add(...)`, same "seed
it yourself" approach `test_web_events.py`/`test_web_notifications.py`
use. The `Feedback` rows these tests create are deliberately left in
place against that real Silver watch (this shared DB is never truncated
between test runs) -- exactly the same precedent those two modules set
for their own `Event`/`Notification` rows, and the reason issue #39
requires extending `tests/test_pipeline_discover.py`'s
`_delete_silver_watch_and_dependents` to also delete `Feedback` rows.
Every assertion against the shared `feedback` table is containment-based,
never exact-count, same pattern `test_web_events.py` establishes.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Category, Event, EventCategory, Feedback, Notification, Watch
from nie.pipeline.score import build_context_bundle
from nie.seed.categories import seed_categories
from nie.seed.run import SILVER_WATCH_SLUG, seed
from nie.web.app import create_app
from nie.web.deps import get_session

REPO_ROOT = Path(__file__).parent.parent

# The exact value set `nie.models.Feedback.verdict`'s `CheckConstraint`
# allows -- the 5 verdicts issue #39 requires round-tripping.
_VERDICTS = ["useful", "not_useful", "too_many_similar", "more_like_this", "less_of_this"]

# Field names of `nie.schemas.FeedbackResponse`, as serialized on the
# wire -- used to check response shape without depending on exact values
# other than the ones each test controls.
_FEEDBACK_RESPONSE_FIELDS = {"id", "watch_id", "event_id", "verdict", "note", "created_at"}


def _is_feedback_response_shaped(value: object) -> bool:
    return isinstance(value, dict) and value.keys() == _FEEDBACK_RESPONSE_FIELDS


@pytest.fixture
def migrated_db() -> None:
    """Run `alembic upgrade head` against the live Compose DB.

    A plain (sync) fixture -- Alembic drives its own event loop
    internally, so this must run outside pytest-asyncio's loop for the
    test function. Re-running this is idempotent.
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


@pytest.fixture
async def silver_watch_id(session_factory: async_sessionmaker[AsyncSession]) -> uuid.UUID:
    """Seed the DB (idempotent) and return the Silver watch's id, for
    tests to attach their own `Event`/`Notification`/`Feedback` rows to
    via `watch_id`.
    """
    async with session_factory() as session:
        await seed(session)
        result = await session.execute(select(Watch.id).where(Watch.slug == SILVER_WATCH_SLUG))
        watch_id: uuid.UUID = result.scalar_one()
        return watch_id


@pytest.fixture
async def seeded_client(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
) -> AsyncIterator[AsyncClient]:
    """A `create_app()` instance with `get_session` overridden to
    `session_factory`, seeded with the Silver watch, driven over
    `ASGITransport` via `httpx.AsyncClient`.
    """
    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


def _unique_title(label: str) -> str:
    return f"{label} {uuid.uuid4()}"


async def _make_event(session: AsyncSession, watch_id: uuid.UUID, *, title: str) -> Event:
    event = Event(
        watch_id=watch_id,
        title=title,
        fact_summary="A fact summary.",
        interpretation="An interpretation.",
        entities=[],
        embedding=[0.0] * 384,
    )
    session.add(event)
    await session.flush()
    return event


async def _seed_event(
    session_factory: async_sessionmaker[AsyncSession], watch_id: uuid.UUID, *, title: str
) -> uuid.UUID:
    async with session_factory() as session:
        event = await _make_event(session, watch_id, title=title)
        await session.commit()
        return event.id


async def test_each_verdict_round_trips_a_feedback_row(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Each of the 5 allowed verdicts, submitted via the HTTP layer,
    inserts exactly one `feedback` row with the correct
    `event_id`/`watch_id`/`verdict`.
    """
    for verdict in _VERDICTS:
        event_id = await _seed_event(
            session_factory, silver_watch_id, title=_unique_title(f"Feedback {verdict}")
        )

        response = await seeded_client.post(f"/events/{event_id}/feedback?verdict={verdict}")
        assert response.status_code == 200
        body = response.json()
        assert _is_feedback_response_shaped(body)
        assert body["event_id"] == str(event_id)
        assert body["watch_id"] == str(silver_watch_id)
        assert body["verdict"] == verdict

        async with session_factory() as session:
            result = await session.execute(
                select(Feedback).where(
                    Feedback.event_id == event_id, Feedback.verdict == verdict
                )
            )
            rows = result.scalars().all()
            assert len(rows) == 1
            assert rows[0].watch_id == silver_watch_id


async def test_invalid_verdict_returns_422(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """An unrecognized verdict value returns `422` with a JSON
    `{"detail": ...}` body, not a `500`, and inserts no row.
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Invalid verdict")
    )

    response = await seeded_client.post(f"/events/{event_id}/feedback?verdict=super_useful")
    assert response.status_code == 422
    assert "detail" in response.json()

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.event_id == event_id))
        assert result.scalars().all() == []


async def test_missing_verdict_returns_422(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Omitting the `verdict` query param entirely returns `422` (FastAPI's
    own required-query-param validation), not a `500`.
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Missing verdict")
    )

    response = await seeded_client.post(f"/events/{event_id}/feedback")
    assert response.status_code == 422


async def test_unknown_event_id_returns_404(seeded_client: AsyncClient) -> None:
    """A `POST` against a non-existent `event_id` returns `404` with a
    JSON `{"detail": ...}` body, not a `500`.
    """
    response = await seeded_client.post(f"/events/{uuid.uuid4()}/feedback?verdict=useful")
    assert response.status_code == 404
    assert "detail" in response.json()


async def test_hx_request_default_target_gets_event_detail_fragment_with_confirmation(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """`HX-Request: true` with no (or an unrecognized) `HX-Target` header
    re-renders `partials/event_detail.html` for the submitting event, and
    the response shows the verdict was recorded.
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("HX event detail")
    )

    response = await seeded_client.post(
        f"/events/{event_id}/feedback?verdict=useful",
        headers={"HX-Request": "true", "HX-Target": "event-detail"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="event-detail">' in response.text
    assert "Feedback recorded: useful" in response.text


async def test_hx_request_notification_list_target_gets_notification_list_fragment(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """`HX-Request: true` with `HX-Target: notification-list` (the inbox's
    per-notification feedback buttons) re-renders `partials/
    notification_list.html` scoped to the feedback's watch, and shows the
    verdict was recorded against the submitting notification's event.
    """
    title = _unique_title("HX notification list")
    async with session_factory() as session:
        event = await _make_event(session, silver_watch_id, title=title)
        notification = Notification(
            watch_id=silver_watch_id,
            event_id=event.id,
            reason="new-event",
            payload={"title": title, "fact_summary": "x", "interpretation": "y"},
            channels_sent=["inapp"],
        )
        session.add(notification)
        await session.commit()
        event_id = event.id

    response = await seeded_client.post(
        f"/events/{event_id}/feedback?verdict=not_useful",
        headers={"HX-Request": "true", "HX-Target": "notification-list"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="notification-list">' in response.text
    assert "Feedback recorded: not_useful" in response.text


async def test_non_blank_note_persists_trimmed(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """A non-blank `note` form field, submitted alongside a verdict, is
    persisted on the `feedback` row with leading/trailing whitespace
    stripped (#50).
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Note non-blank")
    )

    response = await seeded_client.post(
        f"/events/{event_id}/feedback?verdict=useful",
        data={"note": "  Great catch, very relevant  "},
    )
    assert response.status_code == 200
    body = response.json()
    assert _is_feedback_response_shaped(body)
    assert body["note"] == "Great catch, very relevant"

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.event_id == event_id))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].note == "Great catch, very relevant"


async def test_no_note_field_persists_null(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Submitting a verdict with no `note` field at all continues to
    persist `note IS NULL` (regression coverage for #39's existing
    behavior, unaffected by #50).
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Note absent")
    )

    response = await seeded_client.post(f"/events/{event_id}/feedback?verdict=useful")
    assert response.status_code == 200
    body = response.json()
    assert body["note"] is None

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.event_id == event_id))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].note is None


async def test_whitespace_only_note_persists_null(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """A whitespace-only `note` (e.g. `"   "`) persists `note IS NULL`,
    not the literal whitespace (#50).
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Note whitespace-only")
    )

    response = await seeded_client.post(
        f"/events/{event_id}/feedback?verdict=useful",
        data={"note": "   "},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["note"] is None

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.event_id == event_id))
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].note is None


async def test_plain_request_gets_json(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """No `HX-Request` header gets back JSON, a `FeedbackResponse`."""
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Plain JSON")
    )

    response = await seeded_client.post(f"/events/{event_id}/feedback?verdict=useful")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert _is_feedback_response_shaped(response.json())


# ---------------------------------------------------------------------------
# DELETE /events/{event_id}/feedback/{feedback_id} -- withdraw (#51)
# ---------------------------------------------------------------------------


async def test_undo_control_shown_and_undo_clears_event_detail_confirmation(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Submitting feedback from the event detail page shows an "Undo"
    control (`hx-delete` pointed at the just-created row's id) alongside
    the confirmation; clicking it (a `DELETE` with `HX-Request: true`,
    `HX-Target: event-detail`) re-renders `partials/event_detail.html`
    back to its plain, no-confirmation state, and the row is gone from
    the `feedback` table.
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Undo event detail")
    )

    submit_response = await seeded_client.post(
        f"/events/{event_id}/feedback?verdict=useful",
        headers={"HX-Request": "true", "HX-Target": "event-detail"},
    )
    assert submit_response.status_code == 200
    assert "Feedback recorded: useful" in submit_response.text
    assert "Undo" in submit_response.text

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.event_id == event_id))
        feedback = result.scalar_one()

    assert f"/events/{event_id}/feedback/{feedback.id}" in submit_response.text

    undo_response = await seeded_client.delete(
        f"/events/{event_id}/feedback/{feedback.id}",
        headers={"HX-Request": "true", "HX-Target": "event-detail"},
    )
    assert undo_response.status_code == 200
    assert undo_response.headers["content-type"].startswith("text/html")
    assert '<div id="event-detail">' in undo_response.text
    assert "feedback-confirmation" not in undo_response.text
    assert "Feedback recorded" not in undo_response.text

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.id == feedback.id))
        assert result.scalar_one_or_none() is None


async def test_undo_control_shown_and_undo_clears_notification_list_confirmation(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Same "Undo" affordance from the notification inbox
    (`HX-Target: notification-list`): clicking it re-renders
    `partials/notification_list.html` with that notification's
    confirmation cleared, and the row is gone from the `feedback` table.
    """
    title = _unique_title("Undo notification list")
    async with session_factory() as session:
        event = await _make_event(session, silver_watch_id, title=title)
        notification = Notification(
            watch_id=silver_watch_id,
            event_id=event.id,
            reason="new-event",
            payload={"title": title, "fact_summary": "x", "interpretation": "y"},
            channels_sent=["inapp"],
        )
        session.add(notification)
        await session.commit()
        event_id = event.id

    submit_response = await seeded_client.post(
        f"/events/{event_id}/feedback?verdict=not_useful",
        headers={"HX-Request": "true", "HX-Target": "notification-list"},
    )
    assert submit_response.status_code == 200
    assert "Feedback recorded: not_useful" in submit_response.text
    assert "Undo" in submit_response.text

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.event_id == event_id))
        feedback = result.scalar_one()

    assert f"/events/{event_id}/feedback/{feedback.id}" in submit_response.text

    undo_response = await seeded_client.delete(
        f"/events/{event_id}/feedback/{feedback.id}",
        headers={"HX-Request": "true", "HX-Target": "notification-list"},
    )
    assert undo_response.status_code == 200
    assert undo_response.headers["content-type"].startswith("text/html")
    assert '<div id="notification-list">' in undo_response.text
    assert "feedback-confirmation" not in undo_response.text
    assert "Feedback recorded" not in undo_response.text

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.id == feedback.id))
        assert result.scalar_one_or_none() is None


async def test_withdraw_nonexistent_feedback_id_returns_404(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """`DELETE` against a `feedback_id` that doesn't exist at all returns
    `404` with a JSON `{"detail": ...}` body, not a `500`.
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Withdraw nonexistent")
    )

    response = await seeded_client.delete(f"/events/{event_id}/feedback/{uuid.uuid4()}")
    assert response.status_code == 404
    assert "detail" in response.json()


async def test_withdraw_mismatched_event_id_returns_404(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """A `feedback_id` that exists but belongs to a different `event_id`
    than the one in the path returns `404`, same as a nonexistent id --
    and the row is left untouched.
    """
    event_a_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Withdraw mismatch A")
    )
    event_b_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Withdraw mismatch B")
    )

    submit_response = await seeded_client.post(f"/events/{event_a_id}/feedback?verdict=useful")
    assert submit_response.status_code == 200
    feedback_id = uuid.UUID(submit_response.json()["id"])

    response = await seeded_client.delete(f"/events/{event_b_id}/feedback/{feedback_id}")
    assert response.status_code == 404
    assert "detail" in response.json()

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.id == feedback_id))
        assert result.scalar_one_or_none() is not None


async def test_withdraw_plain_request_returns_204_empty_body(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Called directly with no `HX-Request` header on a valid, matching
    `event_id`/`feedback_id`, the endpoint returns `204 No Content` with
    an empty body -- same convention `DELETE /context/{item_id}` (#35)
    already uses -- and the row is actually gone from the `feedback`
    table (hard delete, not a soft delete/flag).
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Withdraw plain 204")
    )
    submit_response = await seeded_client.post(f"/events/{event_id}/feedback?verdict=useful")
    feedback_id = uuid.UUID(submit_response.json()["id"])

    response = await seeded_client.delete(f"/events/{event_id}/feedback/{feedback_id}")
    assert response.status_code == 204
    assert response.content == b""

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.id == feedback_id))
        assert result.scalar_one_or_none() is None


async def test_double_delete_returns_404_not_silent_success(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Withdrawing an already-withdrawn row (double-`DELETE`, or a
    double-click on Undo) returns the same `404` as any other
    nonexistent `feedback_id` -- not a `500`, and not a silent no-op
    success.
    """
    event_id = await _seed_event(
        session_factory, silver_watch_id, title=_unique_title("Double delete")
    )
    submit_response = await seeded_client.post(f"/events/{event_id}/feedback?verdict=useful")
    feedback_id = uuid.UUID(submit_response.json()["id"])

    first = await seeded_client.delete(f"/events/{event_id}/feedback/{feedback_id}")
    assert first.status_code == 204

    second = await seeded_client.delete(f"/events/{event_id}/feedback/{feedback_id}")
    assert second.status_code == 404
    assert "detail" in second.json()


async def test_withdrawn_feedback_not_counted_by_feedback_bucket(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Regression coverage for `FeedbackBucket`
    (`src/nie/pipeline/score.py`, #27/#28): withdrawing a `feedback` row
    via `DELETE` is a hard delete, so a subsequent scoring run's rolling
    feedback summary (`build_context_bundle`) no longer counts it --
    `FeedbackBucket` counts *all* rows in its window (no code change
    needed there, it queries live table state), so this only holds if
    the row is truly gone, not soft-deleted/flagged.
    """
    async with session_factory() as session:
        await seed_categories(session)
        category_result = await session.execute(
            select(Category).where(Category.slug == "market")
        )
        category = category_result.scalar_one()
        watch = await session.get(Watch, silver_watch_id)
        assert watch is not None

        event = await _make_event(
            session, silver_watch_id, title=_unique_title("Withdrawn bucket regression")
        )
        session.add(EventCategory(event_id=event.id, category_id=category.id))
        await session.commit()
        event_id = event.id

    submit_response = await seeded_client.post(f"/events/{event_id}/feedback?verdict=useful")
    assert submit_response.status_code == 200
    feedback_id = uuid.UUID(submit_response.json()["id"])

    async with session_factory() as session:
        before_watch = await session.get(Watch, silver_watch_id)
        assert before_watch is not None
        before_event = await session.get(Event, event_id)
        assert before_event is not None
        bundle_before = await build_context_bundle(session, before_watch, before_event)

    before_count = next(
        (
            bucket.count
            for bucket in bundle_before.feedback_summary
            if bucket.category_slug == "market" and bucket.verdict == "useful"
        ),
        0,
    )
    assert before_count >= 1

    delete_response = await seeded_client.delete(f"/events/{event_id}/feedback/{feedback_id}")
    assert delete_response.status_code == 204

    async with session_factory() as session:
        after_watch = await session.get(Watch, silver_watch_id)
        assert after_watch is not None
        after_event = await session.get(Event, event_id)
        assert after_event is not None
        bundle_after = await build_context_bundle(session, after_watch, after_event)

    after_count = next(
        (
            bucket.count
            for bucket in bundle_after.feedback_summary
            if bucket.category_slug == "market" and bucket.verdict == "useful"
        ),
        0,
    )
    assert after_count == before_count - 1
