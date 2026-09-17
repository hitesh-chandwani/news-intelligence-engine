"""Tests for src/nie/web/routers/notifications.py (issue #38).

Follows the exact `migrated_db`/`session_factory`/`silver_watch_id`/
`seeded_client` fixture pattern `tests/test_web_events.py` (#37)
established: a fresh engine per test (not `nie.db`'s module-level
singleton), `create_app()` + `app.dependency_overrides[get_session]`,
`httpx.AsyncClient` + `ASGITransport` (no real server process), seeded via
`nie.seed.run.seed`.

Since #48 (the stage that would create `Notification` rows via the real
pipeline) isn't wired yet, every test creates its own `Event` +
`Notification` row(s) directly against the real Silver watch via
`session.add(...)`, same "seed it yourself" approach `test_web_events.py`
uses for `Event`. These rows are deliberately left in place (this shared
DB is never truncated between test runs) -- exactly the same precedent
`test_web_events.py` set for its own `Event` rows, and the reason issue
#38 requires extending `tests/test_pipeline_discover.py`'s
`_delete_silver_watch_and_dependents` to also delete `Notification` rows.
Every assertion against the shared `notification` table is
containment-based, never exact-count, same pattern `test_web_events.py`
establishes.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
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
from nie.web.routers import notifications as notifications_router

REPO_ROOT = Path(__file__).parent.parent

# Field names of `nie.schemas.NotificationResponse`, as serialized on the
# wire -- used to check response shape without depending on exact values
# other than the ones each test controls.
_NOTIFICATION_RESPONSE_FIELDS = {
    "id",
    "watch_id",
    "event_id",
    "reason",
    "payload",
    "channels_sent",
    "created_at",
    "read_at",
}


def _is_notification_response_shaped(value: object) -> bool:
    return isinstance(value, dict) and value.keys() == _NOTIFICATION_RESPONSE_FIELDS


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
    """Build a `NotificationPayload`-shaped dict (see
    `src/nie/pipeline/notify.py`) for `payload` -- hand-built here rather
    than via `build_notification_payload` since this test module creates
    its own `Notification` rows directly (#48, the real send path, isn't
    wired yet).
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
        "sources": [
            {
                "title": "Example Article",
                "url": "https://example.com/article",
                "source_name": "Example News",
                "published_at": None,
            }
        ],
    }


async def _make_notification(
    session: AsyncSession,
    watch_id: uuid.UUID,
    event_id: uuid.UUID,
    *,
    payload: dict[str, Any],
    reason: str = "new-event",
) -> Notification:
    notification = Notification(
        watch_id=watch_id,
        event_id=event_id,
        reason=reason,
        payload=payload,
        channels_sent=["inapp"],
    )
    session.add(notification)
    await session.flush()
    return notification


async def _seed_notification(
    session_factory: async_sessionmaker[AsyncSession],
    watch_id: uuid.UUID,
    *,
    title: str,
    payload_overrides: dict[str, Any] | None = None,
) -> Notification:
    async with session_factory() as session:
        event = await _make_event(session, watch_id, title=title)
        payload = _payload(event.id, title)
        if payload_overrides:
            payload.update(payload_overrides)
        notification = await _make_notification(session, watch_id, event.id, payload=payload)
        await session.commit()
        await session.refresh(notification)
        return notification


async def test_mark_read_round_trip(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Posting to `/notifications/{id}/read` for a notification with
    `read_at is None` returns success and, re-fetched, has `read_at` set
    to a non-`None` value.
    """
    notification = await _seed_notification(
        session_factory, silver_watch_id, title=_unique_title("Mark read round trip")
    )
    assert notification.read_at is None

    response = await seeded_client.post(f"/notifications/{notification.id}/read")
    assert response.status_code == 200

    async with session_factory() as session:
        refreshed = await session.get(Notification, notification.id)
        assert refreshed is not None
        assert refreshed.read_at is not None


async def test_mark_read_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """A second `POST` to an already-read notification stays a `200` and
    keeps `read_at` set (mirrors `mark_read`'s own set-once-idempotent
    contract per #33, exercised here through the HTTP layer).
    """
    notification = await _seed_notification(
        session_factory, silver_watch_id, title=_unique_title("Mark read idempotent")
    )

    first = await seeded_client.post(f"/notifications/{notification.id}/read")
    assert first.status_code == 200
    second = await seeded_client.post(f"/notifications/{notification.id}/read")
    assert second.status_code == 200

    async with session_factory() as session:
        refreshed = await session.get(Notification, notification.id)
        assert refreshed is not None
        assert refreshed.read_at is not None


async def test_mark_read_unknown_id_returns_404(seeded_client: AsyncClient) -> None:
    """`POST /notifications/{id}/read` with a random unused UUID returns
    `404` with a JSON `{"detail": ...}` body, and creates/changes no row.
    """
    response = await seeded_client.post(f"/notifications/{uuid.uuid4()}/read")
    assert response.status_code == 404
    assert "detail" in response.json()


async def test_list_notifications_hx_request_gets_html_fragment(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """`HX-Request: true` on `GET /notifications` gets back `text/html`,
    the `partials/notification_list.html` fragment, containing the
    notification this test created.
    """
    title = _unique_title("HX fragment")
    await _seed_notification(session_factory, silver_watch_id, title=title)

    response = await seeded_client.get("/notifications", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="notification-list">' in response.text
    assert title in response.text


async def test_list_notifications_plain_request_gets_json(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """No `HX-Request` header and a non-`text/html` `Accept` (including
    `httpx`'s default `*/*`) gets a JSON array of `NotificationResponse`,
    containing the notification this test created.
    """
    title = _unique_title("Plain JSON")
    notification = await _seed_notification(session_factory, silver_watch_id, title=title)

    response = await seeded_client.get("/notifications")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert isinstance(body, list)
    ids = {row["id"] for row in body}
    assert str(notification.id) in ids
    matching = next(row for row in body if row["id"] == str(notification.id))
    assert _is_notification_response_shaped(matching)
    assert matching["payload"]["title"] == title
    assert matching["read_at"] is None


async def test_list_notifications_html_accept_header_gets_full_page(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """No `HX-Request` header and an `Accept: text/html...` header (a
    plain browser navigation) gets the full `notifications.html` page,
    containing the notification this test created.
    """
    title = _unique_title("Full page")
    await _seed_notification(session_factory, silver_watch_id, title=title)

    response = await seeded_client.get(
        "/notifications", headers={"Accept": "text/html,application/xhtml+xml"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="notification-list">' in response.text
    assert title in response.text


async def test_get_notifications_missing_watch_returns_404(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`GET /notifications` returns `404` with a JSON `{"detail": ...}`
    body when the Silver watch row does not exist -- same "no watch, no
    500" precedent every other router in this module
    (`watch.py`/`context.py`/`preferences.py`/`events.py`) established.

    Same technique `test_web_watch.py`'s `test_missing_silver_watch_returns_404`
    uses: this repo's test DB is shared and never truncated, so actually
    deleting the real Silver watch row is not an option (Postgres enforces
    FK constraints from other tables at `DELETE` time). Instead this
    monkeypatches `nie.web.routers.notifications.SILVER_WATCH_SLUG` to a
    slug that provably doesn't exist, so `_get_silver_watch`'s query
    legitimately finds no row without touching any real data.
    """
    monkeypatch.setattr(
        notifications_router, "SILVER_WATCH_SLUG", "nonexistent-watch-slug-for-404-test"
    )

    async with session_factory() as session:
        await seed(session)

    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/notifications")
    assert response.status_code == 404
    assert "detail" in response.json()


async def test_payload_content_rendered_in_html(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """FR-021 content is rendered straight from `payload`: fact_summary,
    interpretation + importance_rationale, and sources as links to
    `source.url` (FR-022).
    """
    title = _unique_title("Payload content")
    source_url = f"https://example.com/{uuid.uuid4()}"
    await _seed_notification(
        session_factory,
        silver_watch_id,
        title=title,
        payload_overrides={
            "fact_summary": "A specific fact summary.",
            "interpretation": "A specific interpretation.",
            "importance_rationale": "A specific rationale.",
            "sources": [
                {
                    "title": "A Specific Source",
                    "url": source_url,
                    "source_name": "Example News",
                    "published_at": None,
                }
            ],
        },
    )

    response = await seeded_client.get("/notifications", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert "A specific fact summary." in response.text
    assert "A specific interpretation." in response.text
    assert "A specific rationale." in response.text
    assert f'href="{source_url}"' in response.text


async def test_related_events_rendered_when_non_empty(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """`related_events` ("relevant historical context") renders when
    `payload.related_events` is non-empty (FR-021).
    """
    title = _unique_title("Related events")
    related_marker = _unique_title("Historical context event")
    await _seed_notification(
        session_factory,
        silver_watch_id,
        title=title,
        payload_overrides={
            "related_events": [
                {
                    "event_id": str(uuid.uuid4()),
                    "title": related_marker,
                    "event_date": None,
                    "relation": "precedes",
                    "rationale": "happened first",
                }
            ]
        },
    )

    response = await seeded_client.get("/notifications", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert related_marker in response.text


async def test_unread_marker_present_then_absent_after_mark_read(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """The rendered HTML contains an unread marker for a fresh
    notification and does not contain it after that notification is
    marked read (re-`GET` after the `POST`).

    Uses the per-row "Mark read" button's `hx-post` URL
    (`/notifications/{id}/read`) as the checkable marker rather than the
    generic "Unread" badge text: that button only renders on an unread
    row and is scoped to this exact notification's id, so it proves the
    state of *this* row specifically -- a bare "Unread" text search could
    otherwise pass or fail based on unrelated rows left by other tests in
    this shared, never-truncated table.
    """
    notification = await _seed_notification(
        session_factory, silver_watch_id, title=_unique_title("Unread marker")
    )
    mark_read_marker = f"/notifications/{notification.id}/read"

    before = await seeded_client.get("/notifications", headers={"HX-Request": "true"})
    assert before.status_code == 200
    assert mark_read_marker in before.text

    post_response = await seeded_client.post(f"/notifications/{notification.id}/read")
    assert post_response.status_code == 200

    after = await seeded_client.get("/notifications", headers={"HX-Request": "true"})
    assert after.status_code == 200
    assert mark_read_marker not in after.text
