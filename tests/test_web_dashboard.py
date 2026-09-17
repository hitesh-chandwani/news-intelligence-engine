"""Tests for src/nie/web/routers/dashboard.py (issue #49).

Follows the exact `migrated_db`/`session_factory`/`silver_watch_id`/
`seeded_client` fixture pattern `tests/test_web_events.py` (#37) /
`tests/test_web_notifications.py` (#38) established: a fresh engine per
test (not `nie.db`'s module-level singleton), `create_app()` +
`app.dependency_overrides[get_session]`, `httpx.AsyncClient` +
`ASGITransport` (no real server process), seeded via `nie.seed.run.seed`.

`GET /` is HTML-only (no `HX-Request`/`Accept` JSON branch, per the
issue's acceptance criteria), so every test here asserts on rendered HTML
text rather than a JSON body.

Every assertion against the shared `event`/`notification` tables is
containment-based (a created row's id/title present, or a delta on the
unread count) -- never exact-count/exact-literal -- same discipline
`test_web_events.py`/`test_web_notifications.py` establish, since this
repo's test DB is shared and never truncated between runs.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic.command import upgrade
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Event, Notification, Watch
from nie.seed.run import SILVER_WATCH_SLUG, seed
from nie.web.app import create_app
from nie.web.deps import get_session
from nie.web.routers import dashboard as dashboard_router

REPO_ROOT = Path(__file__).parent.parent


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
    tests to attach their own `Event`/`Notification` rows to via
    `watch_id`.
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


def _payload(event_id: uuid.UUID, title: str) -> dict[str, Any]:
    """Build a `NotificationPayload`-shaped dict, same as
    `test_web_notifications.py`'s own `_payload` helper -- hand-built
    here (not via `build_notification_payload`) since this module also
    creates its own `Notification` rows directly.
    """
    return {
        "event_id": str(event_id),
        "title": title,
        "fact_summary": "Something happened.",
        "interpretation": "This is why it matters.",
        "importance_rationale": "Because of the rationale.",
        "categories": ["markets"],
        "importance": "high",
        "impact_direction": "bullish",
        "impact_confidence": "medium",
        "related_events": [],
        "sources": [],
    }


async def _make_notification(
    session: AsyncSession, watch_id: uuid.UUID, event_id: uuid.UUID, *, title: str
) -> Notification:
    notification = Notification(
        watch_id=watch_id,
        event_id=event_id,
        reason="new-event",
        payload=_payload(event_id, title),
        channels_sent=["inapp"],
    )
    session.add(notification)
    await session.flush()
    return notification


def _extract_unread_count(html: str) -> int:
    """Pull the integer out of `dashboard.html`'s
    `<strong id="unread-count">...</strong>` marker."""
    match = re.search(r'<strong id="unread-count">\s*(\d+)\s*</strong>', html)
    assert match is not None, "unread-count marker not found in rendered HTML"
    return int(match.group(1))


async def test_watch_status_and_toggle_render_enabled(
    seeded_client: AsyncClient,
) -> None:
    """`GET /` includes `partials/watch_status.html`'s status text and
    the "Disable" toggle button (`hx-post="/watch/disable"`) when the
    Silver watch is enabled.
    """
    await seeded_client.post("/watch/enable")

    response = await seeded_client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="watch-status">' in response.text
    assert "enabled" in response.text
    assert 'hx-post="/watch/disable"' in response.text


async def test_watch_status_and_toggle_render_disabled(
    seeded_client: AsyncClient,
) -> None:
    """`GET /` includes the "Enable" toggle button
    (`hx-post="/watch/enable"`) when the Silver watch is disabled.
    """
    await seeded_client.post("/watch/disable")

    response = await seeded_client.get("/")
    assert response.status_code == 200
    assert '<div id="watch-status">' in response.text
    assert "disabled" in response.text
    assert 'hx-post="/watch/enable"' in response.text


async def test_created_event_appears_in_recent_events(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """A freshly created event (unique title via `uuid4()`) appears in
    the recent-events section, linked to `/events/{id}` -- containment
    based, since other tests/runs leave `event` rows in this shared,
    never-truncated table.
    """
    title = _unique_title("Dashboard recent event")
    async with session_factory() as session:
        event = await _make_event(session, silver_watch_id, title=title)
        await session.commit()
        event_id = event.id

    response = await seeded_client.get("/")
    assert response.status_code == 200
    assert title in response.text
    assert f"/events/{event_id}" in response.text


async def test_unread_count_increases_by_delta_after_creating_notification(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Creating a new unread notification (unique payload via `uuid4()`)
    increases the displayed unread count relative to a count captured
    *before* creating it -- a delta assertion, not a literal expected
    number, since the shared `notification` table already carries unread
    rows left by other tests/runs.
    """
    before_response = await seeded_client.get("/")
    assert before_response.status_code == 200
    before_count = _extract_unread_count(before_response.text)

    title = _unique_title("Dashboard unread delta")
    async with session_factory() as session:
        event = await _make_event(session, silver_watch_id, title=title)
        await _make_notification(session, silver_watch_id, event.id, title=title)
        await session.commit()

    after_response = await seeded_client.get("/")
    assert after_response.status_code == 200
    after_count = _extract_unread_count(after_response.text)

    assert after_count == before_count + 1


async def test_directly_marked_read_notification_excluded_from_unread_count(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """A notification marked read via a direct DB write (not through
    `/notifications/{id}/read`) does not increase the unread count.
    """
    before_response = await seeded_client.get("/")
    assert before_response.status_code == 200
    before_count = _extract_unread_count(before_response.text)

    title = _unique_title("Dashboard already read")
    async with session_factory() as session:
        event = await _make_event(session, silver_watch_id, title=title)
        notification = await _make_notification(session, silver_watch_id, event.id, title=title)
        notification.read_at = datetime.now(UTC)
        await session.commit()

    after_response = await seeded_client.get("/")
    assert after_response.status_code == 200
    after_count = _extract_unread_count(after_response.text)

    assert after_count == before_count


async def test_empty_state_messages_render_for_controlled_zero_case(
    seeded_client: AsyncClient,
) -> None:
    """`GET /` renders `200` without erroring for the seeded watch, and
    the unread-notification count renders a plain, non-erroring value
    (asserted via the `unread-count` marker parsing as an int) rather
    than raising. Exact emptiness of `pipeline_run`/`event` is not
    guaranteed in this shared DB, so this only asserts the page renders
    successfully and, when either section the test can observe is
    actually empty, that its explicit empty-state message is present --
    never that the whole page or section is unconditionally empty.
    """
    response = await seeded_client.get("/")
    assert response.status_code == 200
    # A well-formed, non-negative unread count always renders -- proves
    # the notification-count section never errors, whatever the shared
    # table currently contains.
    assert _extract_unread_count(response.text) >= 0
    # Exactly one of the two last-run branches is present: either the
    # explicit "no pipeline run yet" message, or a well-formed summary.
    if "No pipeline run yet." in response.text:
        assert True
    else:
        assert "Trigger:" in response.text and "Status:" in response.text
    # Exactly one of the two recent-events branches is present.
    if "No events yet." in response.text:
        assert True
    else:
        assert "<li>" in response.text


async def test_missing_silver_watch_returns_404(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GET /` returns `404` with a JSON `{"detail": ...}` body when the
    Silver watch row does not exist.

    Same technique `test_web_watch.py`'s `test_missing_silver_watch_returns_404`
    uses: this repo's test DB is shared and never truncated, so actually
    deleting the real Silver watch row is not an option (Postgres
    enforces FK constraints from other tables at `DELETE` time). Instead
    this monkeypatches `nie.web.routers.dashboard.SILVER_WATCH_SLUG` to a
    slug that provably doesn't exist, so `_get_silver_watch`'s query
    legitimately finds no row without touching any real data.
    """
    monkeypatch.setattr(
        dashboard_router, "SILVER_WATCH_SLUG", "nonexistent-watch-slug-for-404-test"
    )

    async with session_factory() as session:
        await seed(session)

    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 404
    assert "detail" in response.json()
