"""Tests for src/nie/models.py (the `Watch`, `ContextItem`, `Category` models).

Per _docs/testing-guidelines.md, DB-backed tests run against the real
Docker Compose Postgres (`docker compose up db`), never mocked. This test
runs the Alembic migration chain (issues #4, #5, #6, #7) against that live
database, then exercises `Watch`/`ContextItem`/`Category` through the async
session factory to prove the mapping, unique/check constraints, and foreign
key all work end to end.

The `Category` tests also exercise `nie.seed.categories.seed_categories`
directly (#7) -- `category` is a small global lookup table (unlike `watch`,
it isn't given a fresh per-test-unique slug), so those tests rely on
`seed_categories`'s idempotency to keep the table at exactly the 13 seeded
rows across runs/re-runs, and must not insert any `Category` row with a slug
outside that fixed set.
"""

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Category, ContextItem, Watch
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
