"""Tests for src/nie/scheduler.py + the `lifespan` wiring in
src/nie/web/app.py (issue #40).

Per `_docs/testing-guidelines.md`, DB-backed tests run against the real
Docker Compose Postgres, never mocked, and never make a live network
call. Follows the `migrated_db`/`session_factory` fixture pattern from
`tests/test_pipeline_runner.py`/`tests/test_run.py`: a fresh engine per
test (rather than `nie.db`'s module-level singleton) so pooled asyncpg
connections stay bound to this test's own event loop.

Every test below calls `scheduler_tick` directly -- never the real
`AsyncIOScheduler` interval -- so nothing here depends on real
`POLL_INTERVAL_MINUTES` wall-clock sleeping, per this issue's own test
acceptance criteria.

The "enabled watch" test lets `scheduler_tick` drive a real
`run_pipeline()` call against the live, unmodified `STAGE_REGISTRY` (the
same call the real scheduler makes in production), so it needs the same
no-live-network stubbing `tests/test_pipeline_runner.py`'s end-to-end
test uses: `DISCOVERY_PROVIDERS=stub` (so `discover_stage` makes no RSS
fetch), `trafilatura.fetch_url` stubbed to fail every fetch (so
`extract_stage` makes no live page fetch), and every stage's own
`LLMClient` construction stubbed (so `triage`/`adjudicate`/`synthesize`/
`score`/`relate` make no real LLM API call). This test only asserts the
`pipeline_run` row's `trigger`/`status`/`finished_at` shape -- per-stage
business-logic coverage belongs to each stage's own test module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import nie.scheduler as scheduler_module
from nie.db import create_engine, create_session_factory
from nie.models import PipelineRun, Watch
from nie.pipeline import adjudicate as adjudicate_module
from nie.pipeline import relate as relate_module
from nie.pipeline import score as score_module
from nie.pipeline import synthesize as synthesize_module
from nie.pipeline import triage as triage_module
from nie.pipeline.adjudicate import AdjudicationResult
from nie.pipeline.relate import RelationSet
from nie.pipeline.score import ScoreResult
from nie.pipeline.synthesize import EventRecord
from nie.pipeline.triage import TriageResult
from nie.scheduler import scheduler_tick
from nie.seed.run import SILVER_WATCH_SLUG, seed
from nie.sources import extract_trafilatura
from nie.web.app import create_app
from nie.web.deps import get_session

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


async def _set_watch_status(
    session_factory: async_sessionmaker[AsyncSession], status: str
) -> None:
    """Seed the Silver watch (idempotent) and force its `status` to
    `status`, committing. `seed()`'s own upsert is `ON CONFLICT DO
    NOTHING`, so it never resets `status` on a watch left in a different
    state by an earlier test run against this shared, never-truncated
    test DB -- this helper makes the starting state explicit instead of
    assuming it.
    """
    async with session_factory() as session:
        await seed(session)
        result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
        watch = result.scalar_one()
        watch.status = status
        await session.commit()


async def _count_pipeline_runs(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        result = await session.execute(select(func.count()).select_from(PipelineRun))
        return result.scalar_one()


class _StubTriageClient:
    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[TriageResult]
    ) -> TriageResult:
        return response_model(plausible=True, note="stub triage verdict")


class _StubAdjudicateClient:
    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[AdjudicationResult]
    ) -> AdjudicationResult:
        return response_model(decision="noise", event_id=None, materiality="none")


class _StubSynthesizeClient:
    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[EventRecord]
    ) -> EventRecord:
        return response_model(
            title="Stub Event",
            fact_summary="Stub fact summary.",
            interpretation="Stub interpretation.",
            event_date=None,
            entities=[],
            categories=["other"],
        )


class _StubScoreClient:
    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[ScoreResult]
    ) -> ScoreResult:
        return response_model(
            relevance="irrelevant",
            importance="low",
            impact_direction="neutral",
            impact_reason="Stub score verdict.",
            impact_confidence="low",
        )


class _StubRelateClient:
    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[RelationSet]
    ) -> RelationSet:
        return response_model(relations=[])


async def test_tick_against_disabled_watch_creates_no_pipeline_run(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A tick against a disabled Silver watch is a silent no-op: zero new
    `pipeline_run` rows, no exception.
    """
    await _set_watch_status(session_factory, "disabled")

    before = await _count_pipeline_runs(session_factory)
    await scheduler_tick(session_factory)
    after = await _count_pipeline_runs(session_factory)

    assert after == before


async def test_tick_against_enabled_watch_creates_one_schedule_run(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tick against an enabled Silver watch produces exactly one new
    `pipeline_run` row, with `trigger="schedule"` (the literal string the
    DB `CheckConstraint` requires -- not "scheduled") and a terminal
    `status` (`ok`/`partial`/`failed`).
    """
    monkeypatch.setenv("DISCOVERY_PROVIDERS", "stub")
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", lambda url: None)
    monkeypatch.setattr(triage_module, "LLMClient", lambda *args, **kwargs: _StubTriageClient())
    monkeypatch.setattr(
        adjudicate_module, "LLMClient", lambda *args, **kwargs: _StubAdjudicateClient()
    )
    monkeypatch.setattr(
        synthesize_module, "LLMClient", lambda *args, **kwargs: _StubSynthesizeClient()
    )
    monkeypatch.setattr(score_module, "LLMClient", lambda *args, **kwargs: _StubScoreClient())
    monkeypatch.setattr(relate_module, "LLMClient", lambda *args, **kwargs: _StubRelateClient())

    await _set_watch_status(session_factory, "enabled")

    before = await _count_pipeline_runs(session_factory)
    await scheduler_tick(session_factory)
    after = await _count_pipeline_runs(session_factory)

    assert after == before + 1

    async with session_factory() as session:
        result = await session.execute(
            select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(1)
        )
        latest_run = result.scalar_one()

    assert latest_run.trigger == "schedule"
    assert latest_run.status in {"ok", "partial", "failed"}
    assert latest_run.finished_at is not None


async def test_tick_with_missing_silver_watch_skips_silently(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tick when the Silver watch row doesn't exist (DB migrated but
    never seeded) logs/skips rather than raising, and creates no
    `pipeline_run` row.

    Same technique `tests/test_web_watch.py`'s missing-watch test uses:
    monkeypatch `nie.scheduler.SILVER_WATCH_SLUG` (the module-level
    constant `scheduler_tick` queries `Watch.slug` by) to a slug that
    provably doesn't exist, rather than deleting the real Silver watch
    row from this shared, never-truncated test DB -- Postgres FK
    constraints from other tables make that delete fail outright.
    """
    monkeypatch.setattr(
        scheduler_module, "SILVER_WATCH_SLUG", "nonexistent-watch-slug-for-scheduler-test"
    )

    before = await _count_pipeline_runs(session_factory)
    await scheduler_tick(session_factory)
    after = await _count_pipeline_runs(session_factory)

    assert after == before


async def test_lifespan_starts_and_stops_scheduler_cleanly(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """`create_app()`'s `lifespan` starts the `AsyncIOScheduler` on
    startup and shuts it down cleanly on shutdown, and the app still
    serves requests normally in between -- a broken `lifespan` would
    silently break every other router's tests too, since they all build
    their app via the same `create_app()`.

    `httpx.ASGITransport` (used by every other web test in this suite)
    does not itself send ASGI lifespan events, so this drives
    `app.router.lifespan_context(app)` directly -- the same async context
    manager FastAPI/Starlette invoke internally on real process
    startup/shutdown -- around a request made through the normal
    `ASGITransport` + `AsyncClient` pattern.
    """
    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/watch/status")
        assert response.status_code in {200, 404}
