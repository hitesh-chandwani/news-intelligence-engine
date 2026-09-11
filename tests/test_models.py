"""Tests for src/nie/models.py (the `Watch` model / `watch` table).

Per _docs/testing-guidelines.md, DB-backed tests run against the real
Docker Compose Postgres (`docker compose up db`), never mocked. This test
runs the Alembic migration chain (issues #4, #5) against that live
database, then exercises `Watch` through the async session factory to
prove the mapping, unique constraint, and check constraint all work end
to end.
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
from nie.models import Watch

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
