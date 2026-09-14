"""Tests for src/nie/pipeline/notify.py (issue #30).

`notify_gate` tests are plain synchronous `pytest` functions -- no DB, no
`pytest.mark.asyncio` -- against in-memory, unpersisted `Event`/
`NotificationPreference` instances (the function takes no `session` and
touches no DB per its own contract).

`build_notification_payload` tests are DB-backed per
`_docs/testing-guidelines.md`: they run against the live Compose Postgres,
never mocked. Follows `tests/test_pipeline_relate.py`'s/
`tests/test_pipeline_score.py`'s `migrated_db`/`session_factory` fixture
pattern -- a fresh engine per test, not the module-level singleton, so
pooled asyncpg connections stay bound to this test's own event loop.

Categories are the global, seeded `category` table (13 rows,
`nie.seed.categories.seed_categories`) -- same precedent
`tests/test_pipeline_score.py` set: tests here look up existing seeded
rows (`"market"`/`"price"`) rather than inserting new `Category` rows,
since the Compose Postgres is shared and never truncated between test
runs.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import (
    Category,
    Event,
    EventCategory,
    EventRelation,
    EventSource,
    NotificationPreference,
    Source,
    Watch,
)
from nie.pipeline.notify import build_notification_payload, notify_gate
from nie.seed.categories import seed_categories

REPO_ROOT = Path(__file__).parent.parent

EMBEDDING_DIM = 384
ZERO_EMBEDDING = [0.0] * EMBEDDING_DIM


def unique_slug(prefix: str) -> str:
    """A per-run-unique slug so tests stay independent of prior DB state."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# notify_gate -- plain sync tests, no DB.
# ---------------------------------------------------------------------------


def _event(
    *,
    relevance: str | None = "high",
    importance: str | None = "high",
) -> Event:
    return Event(
        watch_id=uuid.uuid4(),
        title="Some Event",
        fact_summary="Some fact summary.",
        interpretation="Some interpretation.",
        entities=[],
        embedding=ZERO_EMBEDDING,
        relevance=relevance,
        importance=importance,
        impact_direction="neutral",
        impact_reason="Because reasons.",
        impact_confidence="medium",
    )


def _preference(
    *,
    min_importance: str = "medium",
    categories: list[str] | None = None,
) -> NotificationPreference:
    return NotificationPreference(
        watch_id=uuid.uuid4(),
        min_importance=min_importance,
        categories=categories if categories is not None else [],
        channels=[],
    )


def test_notify_gate_false_when_reason_is_none() -> None:
    event = _event()
    preference = _preference()
    assert notify_gate(event, None, ["market"], preference) is False


def test_notify_gate_false_when_relevance_is_irrelevant() -> None:
    event = _event(relevance="irrelevant")
    preference = _preference()
    assert notify_gate(event, "new-event", ["market"], preference) is False


def test_notify_gate_false_when_relevance_is_none() -> None:
    event = _event(relevance=None)
    preference = _preference()
    assert notify_gate(event, "new-event", ["market"], preference) is False


def test_notify_gate_false_when_importance_is_none() -> None:
    event = _event(importance=None)
    preference = _preference()
    assert notify_gate(event, "new-event", ["market"], preference) is False


def test_notify_gate_false_when_importance_below_min_threshold() -> None:
    event = _event(importance="low")
    preference = _preference(min_importance="high")
    assert notify_gate(event, "new-event", ["market"], preference) is False


def test_notify_gate_false_when_categories_disjoint() -> None:
    event = _event()
    preference = _preference(categories=["macro"])
    assert notify_gate(event, "new-event", ["market"], preference) is False


def test_notify_gate_true_when_preference_categories_empty() -> None:
    """`preference.categories == []` means "all categories" -- passes
    regardless of `category_slugs`."""
    event = _event()
    preference = _preference(categories=[])
    assert notify_gate(event, "new-event", ["market"], preference) is True


def test_notify_gate_true_when_categories_overlap() -> None:
    event = _event()
    preference = _preference(categories=["macro", "market"])
    assert notify_gate(event, "new-event", ["market", "price"], preference) is True


def test_notify_gate_true_for_reason_new_event() -> None:
    event = _event(importance="high")
    preference = _preference(min_importance="medium", categories=[])
    assert notify_gate(event, "new-event", ["market"], preference) is True


def test_notify_gate_true_for_reason_material_update() -> None:
    event = _event(importance="high")
    preference = _preference(min_importance="medium", categories=[])
    assert notify_gate(event, "material-update", ["market"], preference) is True


# ---------------------------------------------------------------------------
# build_notification_payload -- DB-backed tests.
# ---------------------------------------------------------------------------


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


async def _make_watch(session: AsyncSession, prefix: str = "notify-test") -> Watch:
    watch = Watch(slug=unique_slug(prefix), name="Notify Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch


def _make_scored_event(
    watch_id: uuid.UUID,
    *,
    title: str = "Some Event",
    event_date: datetime | None = None,
) -> Event:
    return Event(
        watch_id=watch_id,
        title=title,
        fact_summary="A fact summary.",
        interpretation="An interpretation.",
        event_date=event_date,
        entities=["Fed"],
        embedding=ZERO_EMBEDDING,
        relevance="high",
        importance="critical",
        impact_direction="bullish",
        impact_reason="Because of the seeded rationale.",
        impact_confidence="high",
    )


def _make_source(
    watch_id: uuid.UUID,
    *,
    title: str,
    url: str,
    source_name: str,
    discovered_at: datetime,
    published_at: datetime | None = None,
) -> Source:
    return Source(
        watch_id=watch_id,
        url=url,
        title=title,
        source_name=source_name,
        published_at=published_at,
        discovered_at=discovered_at,
        entities=[],
        status="processed",
    )


async def test_build_notification_payload_assembles_every_field_in_order(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime.now(UTC)

    async with session_factory() as session:
        watch = await _make_watch(session)

        await seed_categories(session)
        categories_result = await session.execute(
            select(Category).where(Category.slug.in_(["market", "price"]))
        )
        market, price = sorted(categories_result.scalars(), key=lambda c: c.slug)

        # Three distinct candidates -- different `event_date`s (including a
        # null one) so the `related_events` ORDER BY (event_date desc nulls
        # last, to_event_id asc) is genuinely exercised below, not tied.
        candidate_recent = _make_scored_event(
            watch.id, title="Recent Candidate", event_date=now - timedelta(days=1)
        )
        candidate_old = _make_scored_event(
            watch.id, title="Older Candidate", event_date=now - timedelta(days=3)
        )
        candidate_undated = _make_scored_event(
            watch.id, title="Undated Candidate", event_date=None
        )
        session.add_all([candidate_recent, candidate_old, candidate_undated])
        await session.commit()

        event = _make_scored_event(watch.id, title="Main Event", event_date=now)
        session.add(event)
        await session.commit()

        source_a = _make_source(
            watch.id,
            title="Earlier Source",
            url="https://example.test/earlier",
            source_name="Example Wire",
            discovered_at=now - timedelta(hours=2),
            published_at=now - timedelta(hours=3),
        )
        source_b = _make_source(
            watch.id,
            title="Later Source",
            url="https://example.test/later",
            source_name="Example Times",
            discovered_at=now - timedelta(hours=1),
            published_at=None,
        )
        session.add_all([source_a, source_b])
        await session.commit()

        session.add_all(
            [
                EventCategory(event_id=event.id, category_id=market.id),
                EventCategory(event_id=event.id, category_id=price.id),
                EventSource(event_id=event.id, source_id=source_a.id),
                EventSource(event_id=event.id, source_id=source_b.id),
                EventRelation(
                    from_event_id=event.id,
                    to_event_id=candidate_old.id,
                    relation="precedes",
                    rationale="First relation rationale.",
                ),
                EventRelation(
                    from_event_id=event.id,
                    to_event_id=candidate_recent.id,
                    relation="similar",
                    rationale="Second relation rationale.",
                ),
                EventRelation(
                    from_event_id=event.id,
                    to_event_id=candidate_undated.id,
                    relation="context-for",
                    rationale="Third relation rationale.",
                ),
            ]
        )
        await session.commit()

        payload = await build_notification_payload(session, event)

    assert payload.event_id == event.id
    assert payload.title == "Main Event"
    assert payload.fact_summary == event.fact_summary
    assert payload.interpretation == event.interpretation
    assert payload.importance_rationale == event.impact_reason
    assert payload.importance == "critical"
    assert payload.impact_direction == "bullish"
    assert payload.impact_confidence == "high"

    assert payload.categories == ["market", "price"]

    # sources: ordered by discovered_at ascending, tie-broken by id.
    assert [s.title for s in payload.sources] == ["Earlier Source", "Later Source"]
    assert [s.url for s in payload.sources] == [
        "https://example.test/earlier",
        "https://example.test/later",
    ]
    assert payload.sources[0].source_name == "Example Wire"
    assert payload.sources[0].published_at == source_a.published_at
    assert payload.sources[1].source_name == "Example Times"
    assert payload.sources[1].published_at is None

    # related_events: three distinct candidates with distinct `event_date`s
    # (one null) genuinely exercise the ORDER BY -- event_date descending,
    # NULLS LAST: candidate_recent (now - 1d) first, candidate_old
    # (now - 3d) second, candidate_undated (null) last.
    assert len(payload.related_events) == 3
    assert [r.title for r in payload.related_events] == [
        "Recent Candidate",
        "Older Candidate",
        "Undated Candidate",
    ]
    assert [r.event_id for r in payload.related_events] == [
        candidate_recent.id,
        candidate_old.id,
        candidate_undated.id,
    ]
    assert [r.event_date for r in payload.related_events] == [
        candidate_recent.event_date,
        candidate_old.event_date,
        candidate_undated.event_date,
    ]
    assert [r.relation for r in payload.related_events] == [
        "similar",
        "precedes",
        "context-for",
    ]
    assert [r.rationale for r in payload.related_events] == [
        "Second relation rationale.",
        "First relation rationale.",
        "Third relation rationale.",
    ]


async def test_build_notification_payload_related_events_tie_broken_by_to_event_id(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Two outbound relations whose candidates share the same `event_date`
    must fall back to the second sort key, `to_event_id` ascending -- the
    tie-break the main ordering test above can't exercise since all three
    of its candidates have distinct `event_date`s."""
    now = datetime.now(UTC)

    async with session_factory() as session:
        watch = await _make_watch(session)

        candidate_x = _make_scored_event(watch.id, title="Candidate X", event_date=now)
        candidate_y = _make_scored_event(watch.id, title="Candidate Y", event_date=now)
        session.add_all([candidate_x, candidate_y])
        await session.commit()

        # `id`s are random UUIDs assigned on flush -- sort at runtime rather
        # than assuming which of the two sorts first.
        first, second = sorted([candidate_x, candidate_y], key=lambda c: c.id)

        event = _make_scored_event(watch.id, title="Main Event", event_date=now)
        session.add(event)
        await session.commit()

        session.add_all(
            [
                EventRelation(
                    from_event_id=event.id,
                    to_event_id=candidate_y.id,
                    relation="similar",
                    rationale="Y rationale.",
                ),
                EventRelation(
                    from_event_id=event.id,
                    to_event_id=candidate_x.id,
                    relation="precedes",
                    rationale="X rationale.",
                ),
            ]
        )
        await session.commit()

        payload = await build_notification_payload(session, event)

    assert [r.event_id for r in payload.related_events] == [first.id, second.id]


async def test_build_notification_payload_empty_related_events_when_no_relations(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_scored_event(watch.id, title="Relation-Free Event")
        session.add(event)
        await session.commit()

        payload = await build_notification_payload(session, event)

    assert payload.related_events == []
    assert payload.categories == []
    assert payload.sources == []
