"""Tests for src/nie/models.py (the `Watch`, `ContextItem`, `Category`,
`Source`, `Event`, `EventSource`, `EventRelation`, `EventCategory`,
`NotificationPreference`, `Notification`, `Feedback`, `PipelineRun`
models).

Per _docs/testing-guidelines.md, DB-backed tests run against the real
Docker Compose Postgres (`docker compose up db`), never mocked. This test
runs the Alembic migration chain (issues #4, #5, #6, #7, #8, #9, #10, #42,
#11, #12, #13) against that live database, then exercises `Watch`/
`ContextItem`/`Category`/`Source`/`Event`/`EventSource`/`EventRelation`/
`EventCategory`/`NotificationPreference`/`Notification`/`Feedback`/
`PipelineRun` through the async session factory to prove the mapping,
unique/check constraints, and foreign keys all work end to end.

The `PipelineRun` tests (#13) have no `_seeded_watch` fixture -- unlike
every table before it, `pipeline_run` has no `watch_id` column. It also
has no natural unique key (no `unique_slug`-style helper), so these tests
assert against the row's own returned `id`, never a `COUNT(*)` against
the shared DB, per `_docs/testing-guidelines.md`'s "don't depend on
ordering between tests."

The `Feedback` tests (#12) use a fresh `_seeded_watch`/`_seeded_event` per
test, same as the `NotificationPreference`/`Notification` tests above.

The `NotificationPreference`/`Notification` tests (#11) use a fresh
`_seeded_watch`/`_seeded_event` per test, same as the `Source`/`Event`
tests above, since `notification_preference.watch_id` is a one-row-per-
watch primary key.

The `Category` tests also exercise `nie.seed.categories.seed_categories`
directly (#7) -- `category` is a small global lookup table (unlike `watch`,
it isn't given a fresh per-test-unique slug), so those tests rely on
`seed_categories`'s idempotency to keep the table at exactly the 13 seeded
rows across runs/re-runs, and must not insert any `Category` row with a slug
outside that fixed set.

The `Source` and `Event` tests (#8, #9) use a fixed 384-length embedding
list to match `pgvector.sqlalchemy.Vector(384)` (384 to match fastembed's
`bge-small-en-v1.5`, `design.md` §3, §6). Unlike `Source.embedding`
(nullable), `Event.embedding` is `nullable=False`, so every `Event` test
below supplies one.
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import (
    Category,
    ContextItem,
    Event,
    EventCategory,
    EventRelation,
    EventSource,
    Feedback,
    Notification,
    NotificationPreference,
    PipelineRun,
    Source,
    Watch,
)
from nie.seed.categories import CATEGORIES, seed_categories

REPO_ROOT = Path(__file__).parent.parent


def unique_slug(prefix: str) -> str:
    """A per-run-unique slug so tests stay independent of prior DB state.

    There's no row-cleanup fixture yet (out of scope for #5), so each test
    generates its own slug rather than relying on a fixed one -- keeps
    re-running the suite against the same live DB from colliding with rows
    a previous run left behind.
    """
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def migrated_db() -> None:
    """Run `alembic upgrade head` against the live Compose DB.

    A plain (sync) fixture: Alembic drives its own event loop internally
    (see alembic/env.py), so this must run outside pytest-asyncio's loop
    for the test function. Re-running this is idempotent -- Alembic is a
    no-op when the database is already at the target revision.
    """
    config = Config(str(REPO_ROOT / "alembic.ini"))
    upgrade(config, "head")


@pytest.fixture
async def session_factory(migrated_db: None) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory built from its own engine, scoped to this test.

    `nie.db`'s module-level `engine`/`async_session_factory` is a process
    singleton whose pooled asyncpg connections bind to whichever event loop
    first used them. pytest-asyncio's auto mode gives each test function
    its own loop, so reusing that singleton across tests raises
    ``InterfaceError: another operation is in progress`` on the second
    test. Building a fresh engine (via `nie.db.create_engine`, the same
    factory the app and Alembic use) per test and disposing it afterward
    keeps every connection on the current test's loop.
    """
    engine = create_engine()
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


async def test_create_and_read_back_watch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    slug = unique_slug("silver")
    async with session_factory() as session:
        watch = Watch(slug=slug, name="Silver Commodity", status="enabled")
        session.add(watch)
        await session.commit()
        watch_id = watch.id

    async with session_factory() as session:
        result = await session.execute(select(Watch).where(Watch.id == watch_id))
        fetched = result.scalar_one()

    assert fetched.id is not None
    assert fetched.slug == slug
    assert fetched.name == "Silver Commodity"
    assert fetched.status == "enabled"
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


async def test_duplicate_slug_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    slug = unique_slug("duplicate")
    async with session_factory() as session:
        session.add(Watch(slug=slug, name="First", status="enabled"))
        await session.commit()

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(Watch(slug=slug, name="Second", status="enabled"))
            await session.commit()


async def test_invalid_status_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(Watch(slug=unique_slug("bogus"), name="Bogus", status="bogus"))
            await session.commit()


async def test_create_and_read_back_context_items(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = Watch(slug=unique_slug("silver"), name="Silver Commodity", status="enabled")
        session.add(watch)
        await session.commit()
        watch_id = watch.id

    async with session_factory() as session:
        session.add_all(
            [
                ContextItem(
                    watch_id=watch_id,
                    kind="system",
                    label="Seeded background",
                    body="Silver is a precious and industrial metal.",
                ),
                ContextItem(
                    watch_id=watch_id,
                    kind="user",
                    label="My note",
                    body="Watch for ETF inflows.",
                ),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(ContextItem).where(ContextItem.watch_id == watch_id).order_by(ContextItem.kind)
        )
        items = result.scalars().all()

    assert len(items) == 2
    system_item, user_item = items

    assert system_item.kind == "system"
    assert system_item.label == "Seeded background"
    assert system_item.body == "Silver is a precious and industrial metal."
    assert system_item.created_at is not None
    assert system_item.updated_at is not None

    assert user_item.kind == "user"
    assert user_item.label == "My note"
    assert user_item.body == "Watch for ETF inflows."
    assert user_item.created_at is not None
    assert user_item.updated_at is not None


async def test_invalid_context_item_kind_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = Watch(slug=unique_slug("silver"), name="Silver Commodity", status="enabled")
        session.add(watch)
        await session.commit()
        watch_id = watch.id

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                ContextItem(
                    watch_id=watch_id,
                    kind="bogus",
                    label="Bad kind",
                    body="Should not be inserted.",
                )
            )
            await session.commit()


async def test_context_item_invalid_watch_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                ContextItem(
                    watch_id=uuid.uuid4(),
                    kind="system",
                    label="Orphan",
                    body="No watch references this.",
                )
            )
            await session.commit()


async def test_seed_categories_inserts_all_thirteen_categories(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        await seed_categories(session)

    async with session_factory() as session:
        result = await session.execute(select(Category))
        categories = result.scalars().all()

    assert len(categories) == 13
    assert {(c.slug, c.name) for c in categories} == set(CATEGORIES)


async def test_seed_categories_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        await seed_categories(session)

    async with session_factory() as session:
        await seed_categories(session)

    async with session_factory() as session:
        result = await session.execute(select(Category))
        categories = result.scalars().all()

    assert len(categories) == 13


async def test_duplicate_category_slug_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        await seed_categories(session)

    existing_slug, _ = CATEGORIES[0]

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(Category(slug=existing_slug, name="Duplicate"))
            await session.commit()


async def _seeded_watch(
    session_factory: async_sessionmaker[AsyncSession], prefix: str = "silver"
) -> uuid.UUID:
    """Insert a `Watch` with a fresh unique slug and return its id."""
    async with session_factory() as session:
        watch = Watch(slug=unique_slug(prefix), name="Silver Commodity", status="enabled")
        session.add(watch)
        await session.commit()
        return watch.id


async def test_create_and_read_back_source(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    embedding = [float(i) / 384 for i in range(384)]

    async with session_factory() as session:
        source = Source(
            watch_id=watch_id,
            url="https://example.com/silver-outlook",
            title="Silver Outlook 2026",
            source_name="Example News",
            published_at=datetime(2026, 9, 1, tzinfo=UTC),
            content="Silver prices rose on ETF inflows.",
            extracted_at=datetime(2026, 9, 1, 1, tzinfo=UTC),
            entities=["XAG"],
            embedding=embedding,
            status="extracted",
            triage_note="Relevant to silver ETF flows.",
        )
        session.add(source)
        await session.commit()
        source_id = source.id

    async with session_factory() as session:
        result = await session.execute(select(Source).where(Source.id == source_id))
        fetched = result.scalar_one()

    assert fetched.id is not None
    assert fetched.watch_id == watch_id
    assert fetched.url == "https://example.com/silver-outlook"
    assert fetched.title == "Silver Outlook 2026"
    assert fetched.source_name == "Example News"
    assert fetched.published_at is not None
    assert fetched.discovered_at is not None
    assert fetched.content == "Silver prices rose on ETF inflows."
    assert fetched.extracted_at is not None
    assert fetched.entities == ["XAG"]
    assert fetched.embedding is not None
    assert len(fetched.embedding) == 384
    assert fetched.status == "extracted"
    assert fetched.triage_note == "Relevant to silver ETF flows."


async def test_duplicate_watch_id_url_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    url = f"https://example.com/{unique_slug('duplicate-source')}"

    async with session_factory() as session:
        session.add(
            Source(
                watch_id=watch_id,
                url=url,
                title="First",
                source_name="Example News",
                entities=[],
                status="discovered",
            )
        )
        await session.commit()

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Source(
                    watch_id=watch_id,
                    url=url,
                    title="Second",
                    source_name="Example News",
                    entities=[],
                    status="discovered",
                )
            )
            await session.commit()


async def test_same_url_under_different_watches_succeeds(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id_a = await _seeded_watch(session_factory, "silver")
    watch_id_b = await _seeded_watch(session_factory, "gold")
    url = f"https://example.com/{unique_slug('shared-source')}"

    async with session_factory() as session:
        session.add_all(
            [
                Source(
                    watch_id=watch_id_a,
                    url=url,
                    title="Silver angle",
                    source_name="Example News",
                    entities=[],
                    status="discovered",
                ),
                Source(
                    watch_id=watch_id_b,
                    url=url,
                    title="Gold angle",
                    source_name="Example News",
                    entities=[],
                    status="discovered",
                ),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(select(Source).where(Source.url == url))
        sources = result.scalars().all()

    assert len(sources) == 2
    assert {s.watch_id for s in sources} == {watch_id_a, watch_id_b}


async def test_invalid_source_status_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Source(
                    watch_id=watch_id,
                    url="https://example.com/bogus-status",
                    title="Bogus status",
                    source_name="Example News",
                    entities=[],
                    status="bogus",
                )
            )
            await session.commit()


async def test_source_invalid_watch_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Source(
                    watch_id=uuid.uuid4(),
                    url="https://example.com/orphan-source",
                    title="Orphan",
                    source_name="Example News",
                    entities=[],
                    status="discovered",
                )
            )
            await session.commit()


async def test_create_and_read_back_fully_populated_event(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    slug = unique_slug("event")
    embedding = [float(i) / 384 for i in range(384)]

    async with session_factory() as session:
        event = Event(
            watch_id=watch_id,
            title=f"Silver ETF inflows surge ({slug})",
            fact_summary="ETF holdings rose by 3M ounces this week.",
            interpretation="Investors are rotating into silver as a hedge.",
            event_date=datetime(2026, 9, 1, tzinfo=UTC),
            relevance="high",
            importance="critical",
            impact_direction="bullish",
            impact_reason="Sustained inflows historically precede price rallies.",
            impact_confidence="medium",
            entities=["XAG"],
            embedding=embedding,
        )
        session.add(event)
        await session.commit()
        event_id = event.id

    async with session_factory() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        fetched = result.scalar_one()

    assert fetched.id is not None
    assert fetched.watch_id == watch_id
    assert fetched.title == f"Silver ETF inflows surge ({slug})"
    assert fetched.fact_summary == "ETF holdings rose by 3M ounces this week."
    assert fetched.interpretation == "Investors are rotating into silver as a hedge."
    assert fetched.event_date is not None
    assert fetched.discovered_at is not None
    assert fetched.relevance == "high"
    assert fetched.importance == "critical"
    assert fetched.impact_direction == "bullish"
    assert fetched.impact_reason == "Sustained inflows historically precede price rallies."
    assert fetched.impact_confidence == "medium"
    assert fetched.entities == ["XAG"]
    assert fetched.embedding is not None
    assert len(fetched.embedding) == 384
    assert fetched.last_material_update_at is not None
    assert fetched.created_at is not None
    assert fetched.updated_at is not None


async def test_event_scoring_fields_default_to_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    slug = unique_slug("event")
    embedding = [float(i) / 384 for i in range(384)]

    async with session_factory() as session:
        event = Event(
            watch_id=watch_id,
            title=f"Unscored event ({slug})",
            fact_summary="Central bank announced a policy review.",
            interpretation="Markets are awaiting further clarity.",
            entities=[],
            embedding=embedding,
        )
        session.add(event)
        await session.commit()
        event_id = event.id

    async with session_factory() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        fetched = result.scalar_one()

    assert fetched.relevance is None
    assert fetched.importance is None
    assert fetched.impact_direction is None
    assert fetched.impact_reason is None
    assert fetched.impact_confidence is None


def _event_kwargs(watch_id: uuid.UUID, slug: str) -> dict[str, Any]:
    """Base valid kwargs for constructing an `Event`, so each invalid-value
    test only has to override the one column under test."""
    return {
        "watch_id": watch_id,
        "title": f"Test event ({slug})",
        "fact_summary": "Some observed fact.",
        "interpretation": "Some interpretation.",
        "entities": [],
        "embedding": [float(i) / 384 for i in range(384)],
    }


async def test_invalid_event_relevance_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Event(
                    **_event_kwargs(watch_id, unique_slug("bogus-relevance")),
                    relevance="bogus",
                )
            )
            await session.commit()


async def test_invalid_event_importance_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Event(
                    **_event_kwargs(watch_id, unique_slug("bogus-importance")),
                    importance="bogus",
                )
            )
            await session.commit()


async def test_invalid_event_impact_direction_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Event(
                    **_event_kwargs(watch_id, unique_slug("bogus-impact-direction")),
                    impact_direction="bogus",
                )
            )
            await session.commit()


async def test_invalid_event_impact_confidence_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Event(
                    **_event_kwargs(watch_id, unique_slug("bogus-impact-confidence")),
                    impact_confidence="bogus",
                )
            )
            await session.commit()


async def test_event_invalid_watch_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Event(**_event_kwargs(uuid.uuid4(), unique_slug("orphan-event")))
            )
            await session.commit()


async def _seeded_event(
    session_factory: async_sessionmaker[AsyncSession], watch_id: uuid.UUID, prefix: str = "event"
) -> uuid.UUID:
    """Insert an `Event` with fresh unique-per-run data and return its id."""
    async with session_factory() as session:
        event = Event(**_event_kwargs(watch_id, unique_slug(prefix)))
        session.add(event)
        await session.commit()
        return event.id


async def _seeded_source(
    session_factory: async_sessionmaker[AsyncSession], watch_id: uuid.UUID, prefix: str = "source"
) -> uuid.UUID:
    """Insert a `Source` with fresh unique-per-run data and return its id."""
    async with session_factory() as session:
        source = Source(
            watch_id=watch_id,
            url=f"https://example.com/{unique_slug(prefix)}",
            title="Test source",
            source_name="Example News",
            entities=[],
            status="discovered",
        )
        session.add(source)
        await session.commit()
        return source.id


async def _category_id_by_slug(
    session_factory: async_sessionmaker[AsyncSession], slug: str
) -> uuid.UUID:
    """Ensure the 13 seeded categories exist and return the id for `slug`.

    `EventCategory` tests must not insert `Category` rows of their own --
    per this module's docstring, the `category` table is asserted elsewhere
    to hold exactly the 13 seeded rows, and `seed_categories` is idempotent,
    so calling it here is safe to repeat across tests/runs. Looks up by one
    of the fixed `CATEGORIES` slugs rather than creating new rows.
    """
    async with session_factory() as session:
        await seed_categories(session)

    async with session_factory() as session:
        result = await session.execute(select(Category).where(Category.slug == slug))
        return result.scalar_one().id


async def test_create_and_read_back_event_sources(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)
    source_id_a = await _seeded_source(session_factory, watch_id, "source-a")
    source_id_b = await _seeded_source(session_factory, watch_id, "source-b")

    async with session_factory() as session:
        session.add_all(
            [
                EventSource(event_id=event_id, source_id=source_id_a),
                EventSource(event_id=event_id, source_id=source_id_b),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(EventSource).where(EventSource.event_id == event_id)
        )
        links = result.scalars().all()

    assert {link.source_id for link in links} == {source_id_a, source_id_b}
    assert all(link.linked_at is not None for link in links)


async def test_create_and_read_back_event_relation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    from_event_id = await _seeded_event(session_factory, watch_id, "from-event")
    to_event_id = await _seeded_event(session_factory, watch_id, "to-event")

    async with session_factory() as session:
        session.add(
            EventRelation(
                from_event_id=from_event_id,
                to_event_id=to_event_id,
                relation="precedes",
                rationale="The first event's policy shift caused the second event's rally.",
            )
        )
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(EventRelation).where(EventRelation.from_event_id == from_event_id)
        )
        fetched = result.scalar_one()

    assert fetched.to_event_id == to_event_id
    assert fetched.relation == "precedes"
    assert fetched.rationale == "The first event's policy shift caused the second event's rally."


async def test_duplicate_event_source_pk_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)
    source_id = await _seeded_source(session_factory, watch_id)

    async with session_factory() as session:
        session.add(EventSource(event_id=event_id, source_id=source_id))
        await session.commit()

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(EventSource(event_id=event_id, source_id=source_id))
            await session.commit()


async def test_invalid_event_relation_value_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    from_event_id = await _seeded_event(session_factory, watch_id, "from-event")
    to_event_id = await _seeded_event(session_factory, watch_id, "to-event")

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                EventRelation(
                    from_event_id=from_event_id,
                    to_event_id=to_event_id,
                    relation="bogus",
                    rationale="Not one of the allowed relation values.",
                )
            )
            await session.commit()


async def test_event_relation_self_relation_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                EventRelation(
                    from_event_id=event_id,
                    to_event_id=event_id,
                    relation="similar",
                    rationale="An event cannot be related to itself.",
                )
            )
            await session.commit()


async def test_event_source_orphan_event_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    source_id = await _seeded_source(session_factory, watch_id)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(EventSource(event_id=uuid.uuid4(), source_id=source_id))
            await session.commit()


async def test_event_relation_orphan_from_event_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    to_event_id = await _seeded_event(session_factory, watch_id, "to-event")

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                EventRelation(
                    from_event_id=uuid.uuid4(),
                    to_event_id=to_event_id,
                    relation="similar",
                    rationale="from_event_id does not reference any event.",
                )
            )
            await session.commit()


async def test_create_and_read_back_event_categories(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)
    category_id_a = await _category_id_by_slug(session_factory, CATEGORIES[0][0])
    category_id_b = await _category_id_by_slug(session_factory, CATEGORIES[1][0])

    async with session_factory() as session:
        session.add_all(
            [
                EventCategory(event_id=event_id, category_id=category_id_a),
                EventCategory(event_id=event_id, category_id=category_id_b),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(EventCategory).where(EventCategory.event_id == event_id)
        )
        links = result.scalars().all()

    assert {link.category_id for link in links} == {category_id_a, category_id_b}


async def test_duplicate_event_category_pk_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)
    category_id = await _category_id_by_slug(session_factory, CATEGORIES[0][0])

    async with session_factory() as session:
        session.add(EventCategory(event_id=event_id, category_id=category_id))
        await session.commit()

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(EventCategory(event_id=event_id, category_id=category_id))
            await session.commit()


async def test_notification_preference_default_min_importance_and_array_roundtrip(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    async with session_factory() as session:
        session.add(
            NotificationPreference(
                watch_id=watch_id,
                categories=["markets", "policy"],
                channels=["email", "telegram"],
            )
        )
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(NotificationPreference).where(NotificationPreference.watch_id == watch_id)
        )
        fetched = result.scalar_one()

    assert fetched.min_importance == "medium"
    assert fetched.categories == ["markets", "policy"]
    assert fetched.channels == ["email", "telegram"]


async def test_duplicate_notification_preference_watch_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    async with session_factory() as session:
        session.add(
            NotificationPreference(watch_id=watch_id, categories=[], channels=[])
        )
        await session.commit()

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                NotificationPreference(watch_id=watch_id, categories=[], channels=[])
            )
            await session.commit()


async def test_invalid_notification_preference_min_importance_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                NotificationPreference(
                    watch_id=watch_id,
                    min_importance="bogus",
                    categories=[],
                    channels=[],
                )
            )
            await session.commit()


async def test_notification_preference_orphan_watch_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                NotificationPreference(watch_id=uuid.uuid4(), categories=[], channels=[])
            )
            await session.commit()


async def test_create_and_read_back_fully_populated_notification(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)
    payload = {
        "title": "Silver ETF inflows surge",
        "what_happened": "ETF holdings rose by 3M ounces this week.",
        "why_it_matters": "Investors are rotating into silver as a hedge.",
        "category": "markets",
        "importance": "critical",
        "impact": "bullish",
        "confidence": "medium",
        "historical_context": "Similar inflows preceded the 2020 rally.",
        "sources": ["https://example.com/silver-outlook"],
    }

    async with session_factory() as session:
        notification = Notification(
            watch_id=watch_id,
            event_id=event_id,
            reason="new-event",
            payload=payload,
            channels_sent=["email"],
        )
        session.add(notification)
        await session.commit()
        notification_id = notification.id

    async with session_factory() as session:
        result = await session.execute(
            select(Notification).where(Notification.id == notification_id)
        )
        fetched = result.scalar_one()

    assert fetched.id is not None
    assert fetched.watch_id == watch_id
    assert fetched.event_id == event_id
    assert fetched.reason == "new-event"
    assert fetched.payload == payload
    assert fetched.channels_sent == ["email"]
    assert fetched.created_at is not None
    assert fetched.read_at is None


async def test_invalid_notification_reason_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Notification(
                    watch_id=watch_id,
                    event_id=event_id,
                    reason="bogus",
                    payload={},
                    channels_sent=[],
                )
            )
            await session.commit()


async def test_notification_orphan_watch_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Notification(
                    watch_id=uuid.uuid4(),
                    event_id=event_id,
                    reason="new-event",
                    payload={},
                    channels_sent=[],
                )
            )
            await session.commit()


async def test_notification_orphan_event_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Notification(
                    watch_id=watch_id,
                    event_id=uuid.uuid4(),
                    reason="new-event",
                    payload={},
                    channels_sent=[],
                )
            )
            await session.commit()


FEEDBACK_VERDICTS = [
    "useful",
    "not_useful",
    "too_many_similar",
    "more_like_this",
    "less_of_this",
]


async def test_create_and_read_back_feedback_for_every_verdict(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)

    async with session_factory() as session:
        session.add_all(
            [
                Feedback(
                    watch_id=watch_id,
                    event_id=event_id,
                    verdict=verdict,
                    note=f"Note for {verdict}",
                )
                for verdict in FEEDBACK_VERDICTS
            ]
        )
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(
            select(Feedback).where(Feedback.event_id == event_id)
        )
        rows = result.scalars().all()

    assert len(rows) == len(FEEDBACK_VERDICTS)
    assert {row.verdict for row in rows} == set(FEEDBACK_VERDICTS)
    for row in rows:
        assert row.watch_id == watch_id
        assert row.event_id == event_id
        assert row.note == f"Note for {row.verdict}"
        assert row.created_at is not None


async def test_feedback_without_note_round_trips_as_none(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)

    async with session_factory() as session:
        feedback = Feedback(watch_id=watch_id, event_id=event_id, verdict="useful")
        session.add(feedback)
        await session.commit()
        feedback_id = feedback.id

    async with session_factory() as session:
        result = await session.execute(select(Feedback).where(Feedback.id == feedback_id))
        fetched = result.scalar_one()

    assert fetched.note is None


async def test_invalid_feedback_verdict_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Feedback(watch_id=watch_id, event_id=event_id, verdict="bogus")
            )
            await session.commit()


async def test_feedback_orphan_watch_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    event_id = await _seeded_event(session_factory, watch_id)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Feedback(watch_id=uuid.uuid4(), event_id=event_id, verdict="useful")
            )
            await session.commit()


async def test_feedback_orphan_event_id_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)

    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(
                Feedback(watch_id=watch_id, event_id=uuid.uuid4(), verdict="useful")
            )
            await session.commit()


async def test_pipeline_run_lifecycle_create_running_then_update_to_terminal(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A run is created `running` with empty stats, then updated in place
    (by its own `id`, never by row count) to a terminal status with
    `finished_at` set and a populated `stats` payload."""
    async with session_factory() as session:
        run = PipelineRun(trigger="manual", status="running", stats={})
        session.add(run)
        await session.commit()
        run_id = run.id

    async with session_factory() as session:
        result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
        fetched = result.scalar_one()

    assert fetched.id == run_id
    assert fetched.trigger == "manual"
    assert fetched.status == "running"
    assert fetched.stats == {}
    assert fetched.started_at is not None
    assert fetched.finished_at is None
    assert fetched.error is None

    finished_at = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    stats = {
        "discovered": 5,
        "extracted": 5,
        "new_events": 1,
        "updated_events": 0,
        "notified": 1,
        "errors": 0,
    }

    async with session_factory() as session:
        result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
        row = result.scalar_one()
        row.status = "ok"
        row.finished_at = finished_at
        row.stats = stats
        await session.commit()

    async with session_factory() as session:
        result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
        fetched = result.scalar_one()

    assert fetched.id == run_id
    assert fetched.trigger == "manual"
    assert fetched.status == "ok"
    assert fetched.started_at is not None
    assert fetched.finished_at == finished_at
    assert fetched.stats == stats
    assert fetched.error is None


async def test_invalid_pipeline_run_trigger_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(PipelineRun(trigger="bogus", status="running", stats={}))
            await session.commit()


async def test_invalid_pipeline_run_status_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with pytest.raises(IntegrityError):
        async with session_factory() as session:
            session.add(PipelineRun(trigger="manual", status="bogus", stats={}))
            await session.commit()
