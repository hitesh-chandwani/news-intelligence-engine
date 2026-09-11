"""Tests for src/nie/models.py (the `Watch`, `ContextItem`, `Category`,
`Source` models).

Per _docs/testing-guidelines.md, DB-backed tests run against the real
Docker Compose Postgres (`docker compose up db`), never mocked. This test
runs the Alembic migration chain (issues #4, #5, #6, #7, #8) against that
live database, then exercises `Watch`/`ContextItem`/`Category`/`Source`
through the async session factory to prove the mapping, unique/check
constraints, and foreign key all work end to end.

The `Category` tests also exercise `nie.seed.categories.seed_categories`
directly (#7) -- `category` is a small global lookup table (unlike `watch`,
it isn't given a fresh per-test-unique slug), so those tests rely on
`seed_categories`'s idempotency to keep the table at exactly the 13 seeded
rows across runs/re-runs, and must not insert any `Category` row with a slug
outside that fixed set.

The `Source` tests (#8) use a fixed 384-length embedding list to match
`pgvector.sqlalchemy.Vector(384)` (384 to match fastembed's
`bge-small-en-v1.5`, `design.md` §3, §6).
"""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Category, ContextItem, Source, Watch
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
            entities={"tickers": ["XAG"], "people": []},
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
    assert fetched.entities == {"tickers": ["XAG"], "people": []}
    assert fetched.embedding is not None
    assert len(fetched.embedding) == 384
    assert fetched.status == "extracted"
    assert fetched.triage_note == "Relevant to silver ETF flows."


async def test_duplicate_watch_id_url_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id = await _seeded_watch(session_factory)
    url = "https://example.com/duplicate-source"

    async with session_factory() as session:
        session.add(
            Source(
                watch_id=watch_id,
                url=url,
                title="First",
                source_name="Example News",
                entities={},
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
                    entities={},
                    status="discovered",
                )
            )
            await session.commit()


async def test_same_url_under_different_watches_succeeds(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    watch_id_a = await _seeded_watch(session_factory, "silver")
    watch_id_b = await _seeded_watch(session_factory, "gold")
    url = "https://example.com/shared-source"

    async with session_factory() as session:
        session.add_all(
            [
                Source(
                    watch_id=watch_id_a,
                    url=url,
                    title="Silver angle",
                    source_name="Example News",
                    entities={},
                    status="discovered",
                ),
                Source(
                    watch_id=watch_id_b,
                    url=url,
                    title="Gold angle",
                    source_name="Example News",
                    entities={},
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
                    entities={},
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
                    entities={},
                    status="discovered",
                )
            )
            await session.commit()
