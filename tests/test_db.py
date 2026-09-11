"""Tests for src/nie/db.py.

Per _docs/testing-guidelines.md, DB-backed tests run against the real
Docker Compose Postgres from issue #2 (`docker compose up db`), never
mocked. This test runs the Alembic migration chain (issue #4) against
that live database, then opens a session via `nie.db`'s session factory
and proves the connection is live end to end.
"""

from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import text

from nie.db import async_session_factory

REPO_ROOT = Path(__file__).parent.parent


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


async def test_session_factory_executes_select_1_against_live_db(migrated_db: None) -> None:
    async with async_session_factory() as session:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar_one() == 1
