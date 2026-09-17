"""Tests for src/nie/web/routers/events.py (issue #37).

Follows the exact `migrated_db`/`session_factory`/`seeded_client` fixture
pattern `tests/test_web_watch.py`/`test_web_context.py`/
`test_web_preferences.py` established: a fresh engine per test (not
`nie.db`'s module-level singleton), `create_app()` +
`app.dependency_overrides[get_session]`, `httpx.AsyncClient` +
`ASGITransport` (no real server process), seeded via `nie.seed.run.seed`.

The `event` table has no seed-time rows and no existing factory helper
(unlike `context_item`/`notification_preference`), so every test creates
its own `Event` (+ `Source`/`EventSource`/`EventCategory`/`EventRelation`
as needed) directly against a session opened from `session_factory`, each
with a unique `title` embedding a fresh `uuid4()` so it can't collide with
rows other tests/runs left in this shared, never-truncated DB. Every
assertion against the shared `event` table is containment-based (a
created row's id present/absent by id in the response), never an
exact-count assertion, same pattern `test_web_context.py`/
`test_web_preferences.py` already establish for their own shared tables.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Category, Event, EventCategory, EventRelation, EventSource, Source, Watch
from nie.seed.categories import CATEGORIES
from nie.seed.run import SILVER_WATCH_SLUG, seed
from nie.web.app import create_app
from nie.web.deps import get_session

REPO_ROOT = Path(__file__).parent.parent

# Field names of `nie.schemas.EventSummary`/`EventDetail`, as serialized on
# the wire -- used to check response shape without depending on exact
# values other than the ones each test controls.
_EVENT_SUMMARY_FIELDS = {
    "id",
    "title",
    "event_date",
    "discovered_at",
    "relevance",
    "importance",
    "impact_direction",
    "impact_confidence",
    "categories",
}
_EVENT_DETAIL_FIELDS = {
    "id",
    "title",
    "fact_summary",
    "interpretation",
    "event_date",
    "discovered_at",
    "relevance",
    "importance",
    "impact_direction",
    "impact_reason",
    "impact_confidence",
    "entities",
    "categories",
    "sources",
    "related_events",
}


def _is_event_summary_shaped(value: object) -> bool:
    return isinstance(value, dict) and value.keys() == _EVENT_SUMMARY_FIELDS


def _is_event_detail_shaped(value: object) -> bool:
    return isinstance(value, dict) and value.keys() == _EVENT_DETAIL_FIELDS


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
    tests to attach their own `Event` rows to via `watch_id`.
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


async def _make_event(
    session: AsyncSession,
    watch_id: uuid.UUID,
    *,
    title: str,
    event_date: datetime | None = None,
    relevance: str | None = None,
    importance: str | None = None,
    impact_direction: str | None = None,
    impact_reason: str | None = None,
    impact_confidence: str | None = None,
    fact_summary: str = "A fact summary.",
    interpretation: str = "An interpretation.",
    entities: list[str] | None = None,
) -> Event:
    event = Event(
        watch_id=watch_id,
        title=title,
        fact_summary=fact_summary,
        interpretation=interpretation,
        event_date=event_date,
        relevance=relevance,
        importance=importance,
        impact_direction=impact_direction,
        impact_reason=impact_reason,
        impact_confidence=impact_confidence,
        entities=entities if entities is not None else [],
        embedding=[0.0] * 384,
    )
    session.add(event)
    await session.flush()
    return event


async def _link_category(session: AsyncSession, event_id: uuid.UUID, slug: str) -> None:
    result = await session.execute(select(Category.id).where(Category.slug == slug))
    category_id = result.scalar_one()
    session.add(EventCategory(event_id=event_id, category_id=category_id))


async def _make_source(
    session: AsyncSession, watch_id: uuid.UUID, *, url: str, title: str
) -> Source:
    source = Source(
        watch_id=watch_id,
        url=url,
        title=title,
        source_name="Example News",
        entities=[],
        status="processed",
    )
    session.add(source)
    await session.flush()
    return source


async def test_importance_filter_contains_matching_excludes_others(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """Two events at different `importance` values; `?importance=high`
    contains the high-importance event's id and excludes the low one --
    containment-based, not exact-count (other rows may exist in this
    shared table).
    """
    async with session_factory() as session:
        high = await _make_event(
            session, silver_watch_id, title=_unique_title("High importance"), importance="high"
        )
        low = await _make_event(
            session, silver_watch_id, title=_unique_title("Low importance"), importance="low"
        )
        await session.commit()
        high_id, low_id = high.id, low.id

    response = await seeded_client.get("/events?importance=high")
    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert str(high_id) in ids
    assert str(low_id) not in ids


async def test_category_filter_contains_matching_excludes_unlinked(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """An event linked to a category via `EventCategory` is present when
    filtering by that category's slug; an unlinked event is absent.
    """
    slug = CATEGORIES[0][0]
    async with session_factory() as session:
        linked = await _make_event(session, silver_watch_id, title=_unique_title("Linked"))
        await _link_category(session, linked.id, slug)
        unlinked = await _make_event(session, silver_watch_id, title=_unique_title("Unlinked"))
        await session.commit()
        linked_id, unlinked_id = linked.id, unlinked.id

    response = await seeded_client.get(f"/events?category={slug}")
    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert str(linked_id) in ids
    assert str(unlinked_id) not in ids


async def test_date_range_filter_contains_inside_excludes_outside(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """One event inside a `date_from`/`date_to` range, one outside --
    only the inside one is present in the filtered result.
    """
    now = datetime.now(UTC)
    async with session_factory() as session:
        inside = await _make_event(
            session, silver_watch_id, title=_unique_title("Inside range"), event_date=now
        )
        outside = await _make_event(
            session,
            silver_watch_id,
            title=_unique_title("Outside range"),
            event_date=now - timedelta(days=30),
        )
        await session.commit()
        inside_id, outside_id = inside.id, outside.id

    date_from = (now - timedelta(days=1)).isoformat()
    date_to = (now + timedelta(days=1)).isoformat()
    # `params=` (not an f-string query) so httpx percent-encodes the `+` in
    # the ISO offset -- a literal `+` in a raw query string is decoded as a
    # space (form-encoding convention), which would corrupt the datetime.
    response = await seeded_client.get(
        "/events", params={"date_from": date_from, "date_to": date_to}
    )
    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert str(inside_id) in ids
    assert str(outside_id) not in ids


async def test_relevance_irrelevant_filter_is_not_silently_excluded(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """`relevance=irrelevant` returns a created `relevance="irrelevant"`
    event -- proves it isn't silently excluded (this is a full historical
    record, FR-017/FR-027, not just notification-triggering events).
    """
    async with session_factory() as session:
        event = await _make_event(
            session, silver_watch_id, title=_unique_title("Irrelevant"), relevance="irrelevant"
        )
        await session.commit()
        event_id = event.id

    response = await seeded_client.get("/events?relevance=irrelevant")
    assert response.status_code == 200
    ids = {row["id"] for row in response.json()}
    assert str(event_id) in ids


async def test_null_event_date_present_unfiltered_absent_when_date_filtered(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """An event created with `event_date=None` appears in an unfiltered
    `GET /events` but drops out once any `date_from`/`date_to` filter is
    applied -- standard SQL `NULL` comparison semantics, not a bug.
    """
    async with session_factory() as session:
        event = await _make_event(
            session, silver_watch_id, title=_unique_title("No event date"), event_date=None
        )
        await session.commit()
        event_id = event.id

    unfiltered_response = await seeded_client.get("/events")
    assert unfiltered_response.status_code == 200
    unfiltered_ids = {row["id"] for row in unfiltered_response.json()}
    assert str(event_id) in unfiltered_ids

    now = datetime.now(UTC)
    date_from = (now - timedelta(days=365)).isoformat()
    filtered_response = await seeded_client.get("/events", params={"date_from": date_from})
    assert filtered_response.status_code == 200
    filtered_ids = {row["id"] for row in filtered_response.json()}
    assert str(event_id) not in filtered_ids


async def test_get_event_detail_round_trips_fact_and_interpretation_sources_categories(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """`GET /events/{id}` on a created event round-trips `fact_summary`/
    `interpretation` as distinct fields, and includes its linked source
    (with `url`) and linked category.

    Cleans up the `Source`/`EventSource` rows it creates at the end (the
    only test in this module that creates a `Source` under the *real*
    Silver watch): `tests/test_pipeline_discover.py`'s
    `_delete_silver_watch_and_dependents` bulk-`DELETE`s every `source`
    row for the Silver watch assuming no FK dependents on it -- leaving
    an `event_source` row referencing a `source` this test created would
    break that unrelated, already-passing test the next time it runs
    (Postgres enforces the FK at `DELETE` time). Every other test in this
    module only touches `event`/`event_category`/`event_relation`, which
    that cleanup helper never deletes from, so no such conflict exists
    for them.
    """
    slug = CATEGORIES[1][0]
    async with session_factory() as session:
        event = await _make_event(
            session,
            silver_watch_id,
            title=_unique_title("Detail round-trip"),
            fact_summary="Observed fact only.",
            interpretation="System interpretation only.",
        )
        await _link_category(session, event.id, slug)
        # `source.url` is unique per `(watch_id, url)` -- embed a fresh
        # uuid4() so re-running this test against the shared DB never
        # collides with a URL a previous run already inserted.
        source_url = f"https://example.com/{uuid.uuid4()}"
        source = await _make_source(session, silver_watch_id, url=source_url, title="Article A")
        session.add(EventSource(event_id=event.id, source_id=source.id))
        await session.commit()
        event_id = event.id
        source_id = source.id

    try:
        response = await seeded_client.get(f"/events/{event_id}")
        assert response.status_code == 200
        body = response.json()
        assert _is_event_detail_shaped(body)
        assert body["fact_summary"] == "Observed fact only."
        assert body["interpretation"] == "System interpretation only."
        assert slug in body["categories"]
        assert any(s["url"] == source_url for s in body["sources"])
    finally:
        async with session_factory() as session:
            await session.execute(delete(EventSource).where(EventSource.event_id == event_id))
            await session.execute(delete(Source).where(Source.id == source_id))
            await session.commit()


async def test_get_event_detail_related_events_both_directions(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """An event related to two others (one each direction) shows both in
    `related_events`, with the correct `direction` on each.
    """
    async with session_factory() as session:
        center = await _make_event(session, silver_watch_id, title=_unique_title("Center"))
        successor = await _make_event(session, silver_watch_id, title=_unique_title("Successor"))
        predecessor = await _make_event(
            session, silver_watch_id, title=_unique_title("Predecessor")
        )
        # center precedes successor -> from center's perspective: outgoing.
        session.add(
            EventRelation(
                from_event_id=center.id,
                to_event_id=successor.id,
                relation="precedes",
                rationale="center happened first",
            )
        )
        # predecessor precedes center -> from center's perspective: incoming.
        session.add(
            EventRelation(
                from_event_id=predecessor.id,
                to_event_id=center.id,
                relation="precedes",
                rationale="predecessor happened first",
            )
        )
        await session.commit()
        center_id, successor_id, predecessor_id = center.id, successor.id, predecessor.id

    response = await seeded_client.get(f"/events/{center_id}")
    assert response.status_code == 200
    related = response.json()["related_events"]

    outgoing = [r for r in related if r["direction"] == "outgoing"]
    incoming = [r for r in related if r["direction"] == "incoming"]
    assert any(r["event_id"] == str(successor_id) for r in outgoing)
    assert any(r["event_id"] == str(predecessor_id) for r in incoming)


async def test_get_event_detail_missing_id_returns_404(seeded_client: AsyncClient) -> None:
    """`GET /events/{id}` on a random `uuid4()` returns `404`."""
    response = await seeded_client.get(f"/events/{uuid.uuid4()}")
    assert response.status_code == 404


async def test_invalid_importance_returns_422(seeded_client: AsyncClient) -> None:
    response = await seeded_client.get("/events?importance=urgent")
    assert response.status_code == 422


async def test_invalid_relevance_returns_422(seeded_client: AsyncClient) -> None:
    response = await seeded_client.get("/events?relevance=urgent")
    assert response.status_code == 422


async def test_list_events_hx_request_gets_html_fragment(seeded_client: AsyncClient) -> None:
    """`HX-Request: true` on `GET /events` gets back `text/html`, the
    `partials/event_list.html` fragment.
    """
    response = await seeded_client.get("/events", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="event-list">' in response.text


async def test_list_events_plain_request_gets_json(seeded_client: AsyncClient) -> None:
    """No `HX-Request` header and a non-`text/html` `Accept` (including
    `httpx`'s default `*/*`) gets a JSON array of `EventSummary`.
    """
    response = await seeded_client.get("/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert isinstance(body, list)
    if body:
        assert _is_event_summary_shaped(body[0])


async def test_list_events_html_accept_header_gets_full_page(seeded_client: AsyncClient) -> None:
    """No `HX-Request` header and an `Accept: text/html...` header (a
    plain browser navigation) gets the full `events.html` page.
    """
    response = await seeded_client.get(
        "/events", headers={"Accept": "text/html,application/xhtml+xml"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<form" in response.text
    assert 'name="importance"' in response.text


async def test_get_event_detail_hx_request_gets_html_fragment(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """`HX-Request: true` on `GET /events/{id}` gets back `text/html`,
    the `partials/event_detail.html` fragment.
    """
    async with session_factory() as session:
        event = await _make_event(session, silver_watch_id, title=_unique_title("HX fragment"))
        await session.commit()
        event_id = event.id

    response = await seeded_client.get(f"/events/{event_id}", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="event-detail">' in response.text


async def test_get_event_detail_plain_request_gets_json(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """No `HX-Request` header on `GET /events/{id}` gets back JSON."""
    async with session_factory() as session:
        event = await _make_event(session, silver_watch_id, title=_unique_title("Plain JSON"))
        await session.commit()
        event_id = event.id

    response = await seeded_client.get(f"/events/{event_id}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert _is_event_detail_shaped(response.json())


async def test_get_event_detail_html_accept_header_gets_full_page(
    session_factory: async_sessionmaker[AsyncSession],
    silver_watch_id: uuid.UUID,
    seeded_client: AsyncClient,
) -> None:
    """No `HX-Request` header and an `Accept: text/html...` header gets
    the full `event_detail.html` page, with Fact and Interpretation
    rendered as visually distinct blocks.
    """
    async with session_factory() as session:
        event = await _make_event(
            session,
            silver_watch_id,
            title=_unique_title("Full page"),
            fact_summary="A distinct fact block.",
            interpretation="A distinct interpretation block.",
        )
        await session.commit()
        event_id = event.id

    response = await seeded_client.get(
        f"/events/{event_id}", headers={"Accept": "text/html,application/xhtml+xml"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "A distinct fact block." in response.text
    assert "A distinct interpretation block." in response.text
