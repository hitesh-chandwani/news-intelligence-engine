"""Tests for src/nie/pipeline/runner.py (issue #18).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_run.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test (`nie.db.create_engine`/`create_session_factory`), not the
module-level singleton, so pooled asyncpg connections stay bound to this
test's own event loop.

`discover` (#19) and `extract` (#20) are the first `STAGE_REGISTRY`
entries to become real stages rather than the shared no-op placeholder --
this module's own docstring anticipated exactly this ("each replacing
its own `STAGE_REGISTRY` entry"). `test_all_stages_running_end_to_end_produce_an_ok_run`
below isolates env and seeds the Silver watch so the real
`discover_stage` succeeds without a network call, and only asserts that
it *succeeded* (a `"discovered"` key), leaving the exact insert count to
`tests/test_pipeline_discover.py` -- this file's own concern is the
runner loop, not discover's business logic, and a previous run of the
suite may have already discovered these same fixtures for this
persistent watch. `trafilatura.fetch_url` is monkeypatched to fail every
fetch (same pattern as `tests/test_extract_trafilatura.py`) so that
`extract_stage`, now real too, never makes a live network call against
whatever the real `discover_stage` just inserted -- this test only
asserts `extract`'s stats have the expected shape (per #20's stage
contract), same as `discover`'s, not exact counts.
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
from nie.seed.run import seed
from nie.sources import extract_trafilatura

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


async def test_all_stages_running_end_to_end_produce_an_ok_run(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # discover_stage (#19) builds its own Settings() -- force stub-only
    # discovery so this run makes no network call and doesn't depend on
    # RssProvider, and seed the Silver watch discover_stage looks up.
    monkeypatch.setenv("DISCOVERY_PROVIDERS", "stub")
    # extract_stage (#20) is now real too: whatever discover_stage just
    # inserted for the Silver watch gets picked up in the same run, so
    # trafilatura.fetch_url is monkeypatched to fail every fetch -- this
    # test only cares about the runner loop, not extraction outcomes, and
    # a failed extraction (status="extract_failed") is still a fully
    # valid, non-erroring stage result.
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", lambda url: None)
    async with session_factory() as session:
        await seed(session)

    run_id = await run_pipeline(session_factory, trigger="manual")

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.trigger == "manual"
        assert run.status == "ok"
        assert run.finished_at is not None
        assert run.finished_at >= run.started_at
        assert set(run.stats["discover"]) == {"discovered"}
        assert isinstance(run.stats["discover"]["discovered"], int)
        assert set(run.stats["extract"]) == {"extracted", "extract_failed"}
        assert isinstance(run.stats["extract"]["extracted"], int)
        assert isinstance(run.stats["extract"]["extract_failed"], int)
        for name, _ in STAGE_REGISTRY:
            if name not in ("discover", "extract"):
                assert run.stats[name] == {}


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
