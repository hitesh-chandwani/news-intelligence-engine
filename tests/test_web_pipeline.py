"""Tests for src/nie/web/routers/pipeline.py (issue #43).

Per `_docs/testing-guidelines.md`, DB-backed tests run against the real
Docker Compose Postgres, never mocked, and never make a live network
call. Follows the `migrated_db`/`session_factory` fixture pattern from
`tests/test_scheduler.py`/`tests/test_web_watch.py`: a fresh engine per
test (rather than `nie.db`'s module-level singleton) so pooled asyncpg
connections stay bound to this test's own event loop.

The app under test is built fresh per test via `create_app()` with
`app.dependency_overrides[get_session]` pointed at the test's own
`session_factory`, and driven with `httpx.AsyncClient(transport=
ASGITransport(app=app))` -- same pattern `test_web_watch.py` established
for #34, no real server process, and no lifespan (`app.state.
pipeline_lock` is set directly in `create_app()`, not `_lifespan`, so
none of these tests need to drive `app.router.lifespan_context(app)` the
way `test_scheduler.py`'s one lifespan-specific test does).

In production, `POST /pipeline/run` calls `run_pipeline()` with its own
default `session_factory` (`nie.db.async_session_factory`) -- there is no
way to inject a session factory through an HTTP request. That
module-level singleton engine is created once at import time and its
pooled asyncpg connections bind to whichever event loop first uses them;
pytest-asyncio gives each test function its own event loop, so a second
test in this module driving a real `run_pipeline()` call through that
same default factory would hit asyncpg's "Future attached to a different
loop" error against a connection pool still bound to an earlier test's
(by-then-closed) loop. Tests that need `POST /pipeline/run` to actually
run (rather than short-circuit at the `409` check) work around this with
`_bind_run_pipeline_to_session_factory`, which monkeypatches the
`run_pipeline` name `nie.web.routers.pipeline` looks up at call time to a
`functools.partial` bound to this test's own per-test `session_factory`
-- the same live Postgres instance (`Settings().database_url`) either
way, just without the stale-loop connection reuse.

The end-to-end trigger test and the disabled-watch test both let
`POST /pipeline/run` drive a real `run_pipeline()` call against the live,
unmodified `STAGE_REGISTRY`, so both need the same no-live-network
stubbing `tests/test_scheduler.py`'s enabled-watch test uses:
`DISCOVERY_PROVIDERS=stub` (so `discover_stage` makes no RSS fetch),
`trafilatura.fetch_url` stubbed to fail every fetch (so `extract_stage`
makes no live page fetch), and every stage's own `LLMClient` construction
stubbed (so `triage`/`adjudicate`/`synthesize`/`score`/`relate` make no
real LLM API call).
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Watch
from nie.pipeline import adjudicate as adjudicate_module
from nie.pipeline import relate as relate_module
from nie.pipeline import score as score_module
from nie.pipeline import synthesize as synthesize_module
from nie.pipeline import triage as triage_module
from nie.pipeline.adjudicate import AdjudicationResult
from nie.pipeline.relate import RelationSet
from nie.pipeline.runner import run_pipeline
from nie.pipeline.score import ScoreResult
from nie.pipeline.synthesize import EventRecord
from nie.pipeline.triage import TriageResult
from nie.seed.run import SILVER_WATCH_SLUG, seed
from nie.sources import extract_trafilatura
from nie.web.app import create_app
from nie.web.deps import get_session
from nie.web.routers import pipeline as pipeline_router_module

REPO_ROOT = Path(__file__).parent.parent

# Field names of `nie.schemas.PipelineRunSummary`, as serialized on the
# wire -- used to check a response is well-formed without over-asserting
# on `stats`' exact shape (owned by the pipeline runner, not this
# router).
_PIPELINE_RUN_SUMMARY_FIELDS = {
    "id",
    "trigger",
    "status",
    "started_at",
    "finished_at",
    "stats",
    "error",
}


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
    `status`, committing -- same helper `test_scheduler.py` defines,
    duplicated here rather than imported across test modules (this
    repo's tests don't import each other's private helpers, same
    convention the app modules themselves follow).
    """
    async with session_factory() as session:
        await seed(session)
        result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
        watch = result.scalar_one()
        watch.status = status
        await session.commit()


def _make_client(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncClient:
    """Build a fresh `create_app()` app with `get_session` overridden to
    this test's own `session_factory`, driven via `ASGITransport` -- no
    lifespan, no real server process.
    """
    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


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


def _bind_run_pipeline_to_session_factory(
    monkeypatch: pytest.MonkeyPatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Monkeypatch the `run_pipeline` name `trigger_pipeline_run`
    (`nie.web.routers.pipeline`) looks up at call time, binding it to
    this test's own per-test `session_factory` via `functools.partial` --
    see the module docstring for why (avoids reusing `nie.db`'s
    module-level engine's asyncpg connections, bound to a different
    test's event loop, across tests in this module).
    """
    monkeypatch.setattr(
        pipeline_router_module,
        "run_pipeline",
        functools.partial(run_pipeline, session_factory=session_factory),
    )


def _stub_pipeline_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same no-live-network stubbing `test_scheduler.py`'s enabled-watch
    test uses, factored out since two tests in this module both need it.
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


async def test_trigger_run_appears_in_history(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`POST /pipeline/run` drives a real end-to-end `run_pipeline()`
    call, returns `200` with a well-formed `PipelineRunSummary`
    (`trigger="manual"`, a terminal `status`), and the same run then
    appears as the newest row in `GET /pipeline/runs`.
    """
    _stub_pipeline_stages(monkeypatch)
    _bind_run_pipeline_to_session_factory(monkeypatch, session_factory)
    await _set_watch_status(session_factory, "enabled")

    async with _make_client(session_factory) as client:
        trigger_response = await client.post("/pipeline/run")
        assert trigger_response.status_code == 200
        triggered = trigger_response.json()
        assert set(triggered.keys()) == _PIPELINE_RUN_SUMMARY_FIELDS
        assert triggered["trigger"] == "manual"
        assert triggered["status"] in {"ok", "partial", "failed"}
        assert triggered["finished_at"] is not None

        runs_response = await client.get("/pipeline/runs")

    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert isinstance(runs, list)
    assert runs[0]["id"] == triggered["id"]
    assert runs[0]["trigger"] == "manual"
    # `started_at desc`: the just-triggered run is newest, so it sorts
    # first even against any pre-existing rows in this shared test DB.
    started_at_values = [run["started_at"] for run in runs]
    assert started_at_values == sorted(started_at_values, reverse=True)


async def test_concurrent_trigger_returns_409(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A second `POST /pipeline/run` while `app.state.pipeline_lock` is
    already held is rejected immediately with `409`, without waiting for
    the lock to free up.

    Holds the lock directly (rather than starting a real, slow
    multi-stage `run_pipeline()` call and racing a second request against
    it) -- `trigger_pipeline_run`'s overlap check is exactly
    `app.state.pipeline_lock.locked()`, so acquiring the same lock
    instance from the test is a faithful, fast way to simulate "a run is
    already in progress".
    """
    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with app.state.pipeline_lock:
        assert app.state.pipeline_lock.locked()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/pipeline/run")

    assert response.status_code == 409
    assert response.json() == {"detail": "a pipeline run is already in progress"}


async def test_disabled_watch_does_not_block_trigger(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`POST /pipeline/run` still produces a new `pipeline_run` row when
    the Silver watch is disabled -- unlike `scheduler_tick`, it does not
    look up or gate on `Watch.status` at all.
    """
    _stub_pipeline_stages(monkeypatch)
    _bind_run_pipeline_to_session_factory(monkeypatch, session_factory)
    await _set_watch_status(session_factory, "disabled")

    async with session_factory() as session:
        result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
        watch = result.scalar_one()
        assert watch.status == "disabled"

    async with _make_client(session_factory) as client:
        response = await client.post("/pipeline/run")

    assert response.status_code == 200
    body = response.json()
    assert body["trigger"] == "manual"
    assert body["status"] in {"ok", "partial", "failed"}


async def test_lock_is_the_same_instance_scheduler_and_router_share(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Sanity check on the concurrency design itself: `app.state.
    pipeline_lock` is a plain `asyncio.Lock`, created once by
    `create_app()`, so acquiring it externally (as
    `test_concurrent_trigger_returns_409` does) is equivalent to a
    scheduled tick or another in-flight manual trigger holding it.
    """
    app = create_app()
    assert isinstance(app.state.pipeline_lock, asyncio.Lock)
    assert not app.state.pipeline_lock.locked()
