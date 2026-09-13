"""Tests for src/nie/pipeline/runner.py (issue #18).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_run.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test (`nie.db.create_engine`/`create_session_factory`), not the
module-level singleton, so pooled asyncpg connections stay bound to this
test's own event loop.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import PipelineRun
from nie.pipeline.runner import STAGE_REGISTRY, run_pipeline

REPO_ROOT = Path(__file__).parent.parent


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


async def _succeeding_stage(session: AsyncSession) -> dict[str, int]:
    return {"count": 3}


async def _raising_stage(session: AsyncSession) -> dict[str, int]:
    raise ValueError("boom")


async def test_all_no_op_stages_produce_an_ok_run(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    run_id = await run_pipeline(session_factory, trigger="manual")

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.trigger == "manual"
        assert run.status == "ok"
        assert run.finished_at is not None
        assert run.finished_at >= run.started_at
        assert run.stats == {name: {} for name, _ in STAGE_REGISTRY}


async def test_mixed_stages_produce_a_partial_run_and_keep_running_after_a_failure(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ran_after_failure = False

    async def _stage_after_failure(session: AsyncSession) -> dict[str, int]:
        nonlocal ran_after_failure
        ran_after_failure = True
        return {"after": 1}

    stages = [
        ("succeeds", _succeeding_stage),
        ("fails", _raising_stage),
        ("after", _stage_after_failure),
    ]

    run_id = await run_pipeline(session_factory, trigger="manual", stages=stages)

    assert ran_after_failure is True

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == "partial"
        assert run.stats["succeeds"] == {"count": 3}
        assert run.stats["fails"] == {"error": "boom"}
        assert run.stats["after"] == {"after": 1}


async def test_all_raising_stages_produce_a_failed_run(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stages = [
        ("first", _raising_stage),
        ("second", _raising_stage),
    ]

    run_id = await run_pipeline(session_factory, trigger="manual", stages=stages)

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == "failed"
        assert run.stats["first"] == {"error": "boom"}
        assert run.stats["second"] == {"error": "boom"}
