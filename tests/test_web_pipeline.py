"""Tests for src/nie/web/routers/pipeline.py (issue #43, #53, #54).

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
for #34. Most tests still don't need a real lifespan (`app.state.
pipeline_lock`/`pipeline_tasks` are set directly in `create_app()`, not
`_lifespan`); the one exception is
`test_lifespan_shutdown_cancels_tracked_pipeline_tasks` (#54), which
drives `app.router.lifespan_context(app)` directly, same pattern
`tests/test_scheduler.py`'s own lifespan test uses.

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
way, just without the stale-loop connection reuse. `_bind_run_pipeline_
with_stages` is the #54 sibling of that helper: it additionally overrides
`run_pipeline`'s `stages` parameter with a controllable blocking stage
(`_blocking_stages`, the same `asyncio.Event`-blocked-stage substitute-
for-concurrency pattern `tests/test_pipeline_runner.py`'s
`test_external_cancel_survives_run_pipelines_own_closing_write` uses),
letting a test deterministically observe/control exactly when a
background run is "in flight" -- without it, proving #54's concurrency
invariants (a second trigger's `409` while the first is still running,
real cancellation interrupting later stages, shutdown cancelling a
still-running task, a scheduled tick skipping while a manual trigger
holds the lock) would require racing against a real, fast-finishing
pipeline run.

The end-to-end trigger test(s) and the disabled-watch/client-disconnect
tests let `POST /pipeline/run` drive a real `run_pipeline()` call against
the live, unmodified `STAGE_REGISTRY`, so they need the same
no-live-network stubbing `tests/test_scheduler.py`'s enabled-watch test
uses: `DISCOVERY_PROVIDERS=stub` (so `discover_stage` makes no RSS
fetch), `trafilatura.fetch_url` stubbed to fail every fetch (so
`extract_stage` makes no live page fetch), and every stage's own
`LLMClient` construction stubbed (so `triage`/`adjudicate`/`synthesize`/
`score`/`relate` make no real LLM API call).
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from alembic.command import upgrade
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import PipelineRun, Watch
from nie.pipeline import adjudicate as adjudicate_module
from nie.pipeline import relate as relate_module
from nie.pipeline import score as score_module
from nie.pipeline import synthesize as synthesize_module
from nie.pipeline import triage as triage_module
from nie.pipeline.adjudicate import AdjudicationResult
from nie.pipeline.relate import RelationSet
from nie.pipeline.runner import StageFn, run_pipeline
from nie.pipeline.score import ScoreResult
from nie.pipeline.synthesize import EventRecord
from nie.pipeline.triage import TriageResult
from nie.scheduler import scheduler_tick
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
    monkeypatch.setattr(
        extract_trafilatura.trafilatura, "fetch_url", lambda url, config=None: None
    )
    monkeypatch.setattr(triage_module, "LLMClient", lambda *args, **kwargs: _StubTriageClient())
    monkeypatch.setattr(
        adjudicate_module, "LLMClient", lambda *args, **kwargs: _StubAdjudicateClient()
    )
    monkeypatch.setattr(
        synthesize_module, "LLMClient", lambda *args, **kwargs: _StubSynthesizeClient()
    )
    monkeypatch.setattr(score_module, "LLMClient", lambda *args, **kwargs: _StubScoreClient())
    monkeypatch.setattr(relate_module, "LLMClient", lambda *args, **kwargs: _StubRelateClient())


def _blocking_stages() -> tuple[list[tuple[str, StageFn]], asyncio.Event, asyncio.Event, list[str]]:
    """Build a controllable two-stage `run_pipeline` `stages` list for
    #54's concurrency tests: `"blocking"` sets `stage_entered` then
    awaits `release_stage` before returning (so a test can deterministically
    observe "the background run is now in flight, mid-stage" and either
    release it or cancel the task while it's suspended there); `"after"`
    appends its name to `ran` and returns immediately, so a test can
    assert whether it ever ran at all -- the proof that a cancelled run's
    later stages never execute (issue #54's acceptance criterion), as
    opposed to merely the DB row reading `"cancelled"`.

    Same `asyncio.Event`-blocked-stage substitute for real concurrency
    `tests/test_pipeline_runner.py`'s
    `test_external_cancel_survives_run_pipelines_own_closing_write` uses,
    reused/extended here at the HTTP-router layer.
    """
    stage_entered = asyncio.Event()
    release_stage = asyncio.Event()
    ran: list[str] = []

    async def _blocking(session: AsyncSession) -> dict[str, int]:
        stage_entered.set()
        await release_stage.wait()
        return {"blocked": 1}

    async def _after(session: AsyncSession) -> dict[str, int]:
        ran.append("after")
        return {"after": 1}

    stages: list[tuple[str, StageFn]] = [("blocking", _blocking), ("after", _after)]
    return stages, stage_entered, release_stage, ran


def _bind_run_pipeline_with_stages(
    monkeypatch: pytest.MonkeyPatch,
    session_factory: async_sessionmaker[AsyncSession],
    stages: list[tuple[str, StageFn]],
) -> None:
    """#54 sibling of `_bind_run_pipeline_to_session_factory`: also
    overrides `run_pipeline`'s `stages` parameter (normally the live
    `STAGE_REGISTRY`) with a caller-supplied one, typically
    `_blocking_stages`' controllable stage list.
    """
    monkeypatch.setattr(
        pipeline_router_module,
        "run_pipeline",
        functools.partial(run_pipeline, session_factory=session_factory, stages=stages),
    )


async def _poll_until_terminal(client: AsyncClient, run_id: str) -> dict[str, Any]:
    """Poll `GET /pipeline/runs/{run_id}` until its `status` leaves
    `"running"`. Callers wrap this in `asyncio.wait_for(..., timeout=...)`
    -- per `_docs/testing-guidelines.md`/issue #54's own acceptance
    criteria, this is never used as an untimed loop on its own.
    """
    while True:
        response = await client.get(f"/pipeline/runs/{run_id}")
        assert response.status_code == 200
        body: dict[str, Any] = response.json()
        if body["status"] != "running":
            return body
        await asyncio.sleep(0.05)


async def _count_pipeline_runs(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        result = await session.execute(select(func.count()).select_from(PipelineRun))
        return result.scalar_one()


async def test_trigger_run_returns_202_immediately_then_reaches_terminal_status(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`POST /pipeline/run` returns `202` immediately (#54) -- not after
    the run finishes -- with the `pipeline_run` row already inserted
    (`status="running"`, `finished_at=None`) and serialized as a
    `PipelineRunSummary`. A caller polling `GET /pipeline/runs/{run_id}`
    (new, #54) then observes the status transition from `"running"` to a
    terminal value once the real, unmodified `STAGE_REGISTRY` background
    task completes, and the finished run appears as the newest row in
    `GET /pipeline/runs`.
    """
    _stub_pipeline_stages(monkeypatch)
    _bind_run_pipeline_to_session_factory(monkeypatch, session_factory)
    await _set_watch_status(session_factory, "enabled")

    async with _make_client(session_factory) as client:
        trigger_response = await client.post("/pipeline/run")
        assert trigger_response.status_code == 202
        triggered = trigger_response.json()
        assert set(triggered.keys()) == _PIPELINE_RUN_SUMMARY_FIELDS
        assert triggered["trigger"] == "manual"
        assert triggered["status"] == "running"
        assert triggered["finished_at"] is None
        run_id = triggered["id"]

        finished = await asyncio.wait_for(_poll_until_terminal(client, run_id), timeout=60)
        assert finished["status"] in {"ok", "partial", "failed"}
        assert finished["finished_at"] is not None

        runs_response = await client.get("/pipeline/runs")

    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert isinstance(runs, list)
    assert runs[0]["id"] == run_id
    assert runs[0]["trigger"] == "manual"
    assert runs[0]["status"] == finished["status"]
    # `started_at desc`: the just-triggered run is newest, so it sorts
    # first even against any pre-existing rows in this shared test DB.
    started_at_values = [run["started_at"] for run in runs]
    assert started_at_values == sorted(started_at_values, reverse=True)


async def test_get_single_run_returns_404_for_unknown_id(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """`GET /pipeline/runs/{run_id}` (#54) returns `404` for an id with
    no matching row, mirroring `POST /pipeline/runs/{run_id}/cancel`'s
    own 404 precedent.
    """
    async with _make_client(session_factory) as client:
        response = await client.get(f"/pipeline/runs/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_second_trigger_returns_409_while_first_run_is_in_flight(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves #54's invariant #1 end to end through two real `POST
    /pipeline/run` calls (unlike `test_concurrent_trigger_returns_409`,
    which stands in for "a run is in progress" by holding the lock
    directly): the lock-then-create-row-then-spawn sequence happens
    synchronously in the request handler, so a second concurrent trigger
    is rejected `409` immediately while the first run's background task
    is still executing (blocked mid-stage) -- proving the lock is
    genuinely held for the run's entire background execution, acquired
    before the task is spawned and the handler returns, not lazily
    inside the background task itself.
    """
    stages, stage_entered, release_stage, ran = _blocking_stages()
    _bind_run_pipeline_with_stages(monkeypatch, session_factory, stages)

    async with _make_client(session_factory) as client:
        first_response = await client.post("/pipeline/run")
        assert first_response.status_code == 202
        await asyncio.wait_for(stage_entered.wait(), timeout=5)

        second_response = await client.post("/pipeline/run")
        assert second_response.status_code == 409
        assert second_response.json() == {"detail": "a pipeline run is already in progress"}

        release_stage.set()
        first_run_id = first_response.json()["id"]
        finished = await asyncio.wait_for(
            _poll_until_terminal(client, first_run_id), timeout=10
        )

    assert finished["status"] == "ok"
    assert ran == ["after"]


async def test_cancel_interrupts_the_real_task_before_later_stages_run(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extends `tests/test_pipeline_runner.py`'s
    `test_external_cancel_survives_run_pipelines_own_closing_write`
    blocked-stage pattern to prove #54's real interruption, not just
    #53's DB write: cancelling a run whose background task is blocked
    mid-stage actually raises `asyncio.CancelledError` inside that task
    (`run_pipeline`'s stage loop only catches `Exception`, which does not
    catch `CancelledError`) -- proven here two ways: (1) the stage
    *after* the blocked one never runs and never records its own
    counter, and (2) the task itself finishes and is popped from
    `app.state.pipeline_tasks` promptly (bounded by a short
    `asyncio.wait_for`), rather than only whenever the blocked stage
    eventually returns on its own -- the release event for the blocking
    stage is deliberately never set in this test.
    """
    stages, stage_entered, release_stage, ran = _blocking_stages()
    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session
    _bind_run_pipeline_with_stages(monkeypatch, session_factory, stages)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        trigger_response = await client.post("/pipeline/run")
        assert trigger_response.status_code == 202
        run_id = uuid.UUID(trigger_response.json()["id"])

        await asyncio.wait_for(stage_entered.wait(), timeout=5)
        task = app.state.pipeline_tasks[run_id]
        assert not task.done()

        cancel_response = await client.post(f"/pipeline/runs/{run_id}/cancel")
        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"

        # The task must actually finish promptly -- proving real
        # interruption, not merely a DB flip with the original coroutine
        # left running orphaned in the background (#53's pre-#54
        # behavior). `release_stage` is never set, so this can only
        # complete via cancellation reaching the blocked `await`.
        await asyncio.wait_for(task, timeout=5)

    assert ran == []  # the "after" stage never ran
    assert run_id not in app.state.pipeline_tasks
    assert not app.state.pipeline_lock.locked()

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == "cancelled"
        assert run.error == pipeline_router_module.CANCELLED_ERROR


async def test_lifespan_shutdown_cancels_tracked_pipeline_tasks(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On graceful app shutdown, `_lifespan`'s `finally`
    (`src/nie/web/app.py`) cancels every task still in `app.state.
    pipeline_tasks` (#54) so an in-flight background run is never
    silently dropped or GC-warned, and shutdown itself does not hang
    waiting on it -- bounded here by the outer `asyncio.wait_for`, since
    the blocked stage only needs to notice cancellation at its current
    `await`, which happens promptly.

    Drives a real `app.router.lifespan_context(app)`, same pattern
    `tests/test_scheduler.py`'s own lifespan test uses -- `ASGITransport`
    alone never fires ASGI lifespan events, so `_lifespan`'s shutdown path
    would otherwise never run in a test.
    """
    stages, stage_entered, release_stage, ran = _blocking_stages()
    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session
    _bind_run_pipeline_with_stages(monkeypatch, session_factory, stages)

    async def _run_and_shut_down() -> uuid.UUID:
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/pipeline/run")
                assert response.status_code == 202
                run_id = uuid.UUID(response.json()["id"])

            await asyncio.wait_for(stage_entered.wait(), timeout=5)
            assert run_id in app.state.pipeline_tasks
            # Exiting this `async with` block runs `_lifespan`'s shutdown
            # path, which cancels every task still tracked here.
        return run_id

    run_id = await asyncio.wait_for(_run_and_shut_down(), timeout=10)

    assert run_id not in app.state.pipeline_tasks
    assert ran == []


async def test_scheduler_tick_skips_while_manual_trigger_is_in_flight(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`scheduler_tick` (`nie.scheduler`, unchanged by #54) and `POST
    /pipeline/run` remain mutually exclusive under #54's background
    execution: while a manual trigger's background task is still holding
    `app.state.pipeline_lock` (blocked mid-stage, not yet finished), a
    scheduler tick against the enabled Silver watch skips silently --
    proving `pipeline_lock` stays held for the run's entire background
    execution (not just the synchronous portion of the request), so a
    scheduled tick and a manual trigger still cannot run concurrently.
    """
    stages, stage_entered, release_stage, ran = _blocking_stages()
    _bind_run_pipeline_with_stages(monkeypatch, session_factory, stages)
    await _set_watch_status(session_factory, "enabled")

    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        trigger_response = await client.post("/pipeline/run")
        assert trigger_response.status_code == 202
        await asyncio.wait_for(stage_entered.wait(), timeout=5)

        before = await _count_pipeline_runs(session_factory)
        await asyncio.wait_for(
            scheduler_tick(session_factory, lock=app.state.pipeline_lock), timeout=5
        )
        after = await _count_pipeline_runs(session_factory)
        assert after == before  # the tick skipped: pipeline_lock was still held

        release_stage.set()
        run_id = trigger_response.json()["id"]
        finished = await asyncio.wait_for(_poll_until_terminal(client, run_id), timeout=10)

    assert finished["status"] == "ok"
    assert ran == ["after"]


async def test_client_disconnect_after_202_does_not_affect_the_run(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client that stops listening (closes its `AsyncClient`)
    immediately after receiving `202` has no effect on the in-flight run
    (#54): the background task is tracked only via `app.state`, not tied
    to the request/response cycle or the client connection, so it keeps
    running to completion and reaches a terminal DB status regardless.
    """
    _stub_pipeline_stages(monkeypatch)
    _bind_run_pipeline_to_session_factory(monkeypatch, session_factory)
    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/pipeline/run")
        assert response.status_code == 202
        run_id = uuid.UUID(response.json()["id"])
        task = app.state.pipeline_tasks[run_id]
    # The client's `async with` block has now exited (connection closed)
    # -- the task below, tracked only via `app.state`, is untouched by
    # this: it is not cancelled, and it is not the client that keeps it
    # alive.

    await asyncio.wait_for(task, timeout=60)

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status in {"ok", "partial", "failed"}
        assert run.finished_at is not None


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


async def test_hx_request_trigger_success_swaps_fragment_with_new_running_row(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#62: `POST /pipeline/run` with `HX-Request: true` returns `200
    text/html`, the re-rendered `partials/pipeline_runs.html` fragment,
    showing the newly created row in `"running"` status with a Cancel
    button -- the same two-way negotiation #59 gave `cancel_pipeline_run`.
    Binds a controllable blocking stage (`_blocking_stages`) so the
    background run is deterministically still `"running"` at response
    time, rather than racing a real fast-finishing run.

    Checks for `id="pipeline-runs"` (not the full, exact opening tag)
    since #63 now also appends polling attributes to that same tag
    whenever -- as here -- a rendered row is `"running"`.
    """
    stages, stage_entered, release_stage, ran = _blocking_stages()
    _bind_run_pipeline_with_stages(monkeypatch, session_factory, stages)

    async with _make_client(session_factory) as client:
        response = await client.post("/pipeline/run", headers={"HX-Request": "true"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert 'id="pipeline-runs"' in response.text

        async with session_factory() as session:
            result = await session.execute(
                select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(1)
            )
            newest_run = result.scalar_one()

        assert newest_run.status == "running"
        assert f'hx-post="/pipeline/runs/{newest_run.id}/cancel"' in response.text
        assert "Status: <strong>running</strong>" in response.text

        await asyncio.wait_for(stage_entered.wait(), timeout=5)
        release_stage.set()
        finished = await asyncio.wait_for(
            _poll_until_terminal(client, str(newest_run.id)), timeout=10
        )

    assert finished["status"] == "ok"
    assert ran == ["after"]


async def test_hx_request_trigger_while_already_running_renders_fragment_not_raw_error(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#62: a second `POST /pipeline/run` with `HX-Request: true` while
    `app.state.pipeline_lock` is already held (the `409` race) still
    responds `200 text/html` with the re-rendered fragment -- never a
    raw/broken error -- showing the currently-running row's real state,
    mirroring #59's Cancel `409`/`404` HTMX handling.

    Checks for `id="pipeline-runs"` (not the full, exact opening tag)
    since #63 now also appends polling attributes to that same tag
    whenever -- as here -- a rendered row is `"running"`.
    """
    stages, stage_entered, release_stage, ran = _blocking_stages()
    _bind_run_pipeline_with_stages(monkeypatch, session_factory, stages)

    async with _make_client(session_factory) as client:
        first_response = await client.post("/pipeline/run")
        assert first_response.status_code == 202
        first_run_id = first_response.json()["id"]
        await asyncio.wait_for(stage_entered.wait(), timeout=5)

        second_response = await client.post("/pipeline/run", headers={"HX-Request": "true"})
        assert second_response.status_code == 200
        assert second_response.headers["content-type"].startswith("text/html")
        assert 'id="pipeline-runs"' in second_response.text
        assert f'hx-post="/pipeline/runs/{first_run_id}/cancel"' in second_response.text
        assert "Status: <strong>running</strong>" in second_response.text

        release_stage.set()
        finished = await asyncio.wait_for(
            _poll_until_terminal(client, first_run_id), timeout=10
        )

    assert finished["status"] == "ok"
    assert ran == ["after"]


async def test_disabled_watch_does_not_block_trigger(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`POST /pipeline/run` still produces a new `pipeline_run` row when
    the Silver watch is disabled -- unlike `scheduler_tick`, it does not
    look up or gate on `Watch.status` at all. Also confirms the
    background task still runs the disabled-watch trigger to a terminal
    status (#54), not just that the initial `202` was accepted.
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
        assert response.status_code == 202
        body = response.json()
        assert body["trigger"] == "manual"
        assert body["status"] == "running"
        assert body["finished_at"] is None

        finished = await asyncio.wait_for(
            _poll_until_terminal(client, body["id"]), timeout=60
        )

    assert finished["status"] in {"ok", "partial", "failed"}


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


async def _insert_pipeline_run(
    session_factory: async_sessionmaker[AsyncSession],
    status: str,
    *,
    finished_at: datetime | None = None,
    error: str | None = None,
) -> uuid.UUID:
    """Insert a `pipeline_run` row directly in the DB with no real
    in-flight `run_pipeline()` coroutine behind it -- the "simpler,
    valid case per `_docs/testing-guidelines.md`" the #53 issue calls
    for, as distinct from `test_pipeline_runner.py`'s
    `test_external_cancel_survives_run_pipelines_own_closing_write`
    which drives a real (stubbed) in-flight coroutine.
    """
    async with session_factory() as session:
        run = PipelineRun(
            trigger="manual", status=status, stats={}, finished_at=finished_at, error=error
        )
        session.add(run)
        await session.commit()
        return run.id


async def test_cancel_running_run_returns_200_cancels_row_and_releases_lock(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """`POST /pipeline/runs/{run_id}/cancel` on a `status="running"` row
    (no real in-flight coroutine, just the row) returns `200`, sets
    `status="cancelled"`/`finished_at`/`error`, and force-releases
    `app.state.pipeline_lock` even though this test -- standing in for
    whatever originally acquired it -- never releases it itself.

    Also proves #54's "DB-only cancel when no task is tracked" invariant:
    `run_id` has no entry in `app.state.pipeline_tasks` at all here (it
    was inserted directly by `_insert_pipeline_run`, never via `POST
    /pipeline/run`), so `cancel_pipeline_run`'s `.get(run_id)` lookup
    returns `None` and the `if task is not None: task.cancel()` branch is
    skipped entirely -- this must not raise, and the cancel must still
    succeed exactly as it did before #54.
    """
    run_id = await _insert_pipeline_run(session_factory, "running")

    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session
    assert run_id not in app.state.pipeline_tasks

    lock: asyncio.Lock = app.state.pipeline_lock
    await lock.acquire()
    try:
        assert lock.locked()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/pipeline/runs/{run_id}/cancel")
    finally:
        # The endpoint itself is expected to have already force-released
        # the lock on a successful cancel -- guard the same way it does
        # rather than double-releasing an already-unlocked asyncio.Lock
        # (which raises).
        if lock.locked():
            lock.release()

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == _PIPELINE_RUN_SUMMARY_FIELDS
    assert body["id"] == str(run_id)
    assert body["status"] == "cancelled"
    assert body["error"] == pipeline_router_module.CANCELLED_ERROR
    assert body["finished_at"] is not None
    assert not lock.locked()

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == "cancelled"
        assert run.error == pipeline_router_module.CANCELLED_ERROR
        assert run.finished_at is not None


@pytest.mark.parametrize("status", ["ok", "partial", "failed", "cancelled"])
async def test_cancel_terminal_run_returns_409_and_does_not_modify_row(
    session_factory: async_sessionmaker[AsyncSession],
    status: str,
) -> None:
    """A run whose DB `status` is already terminal -- including already
    `"cancelled"` -- is rejected with `409`, not silently no-op'd, and
    the row is left exactly as it was.
    """
    finished_at = datetime.now(UTC)
    run_id = await _insert_pipeline_run(
        session_factory, status, finished_at=finished_at, error="pre-existing"
    )

    async with _make_client(session_factory) as client:
        response = await client.post(f"/pipeline/runs/{run_id}/cancel")

    assert response.status_code == 409
    assert "not running" in response.json()["detail"]

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == status
        assert run.error == "pre-existing"
        assert run.finished_at == finished_at


async def test_cancel_unknown_run_id_returns_404(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with _make_client(session_factory) as client:
        response = await client.post(f"/pipeline/runs/{uuid.uuid4()}/cancel")

    assert response.status_code == 404


async def test_hx_request_cancel_success_swaps_fragment_to_cancelled_no_button(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#59: `POST /pipeline/runs/{run_id}/cancel` with `HX-Request: true`
    on a `status="running"` row returns `200 text/html`, the re-rendered
    `partials/pipeline_runs.html` fragment (`#pipeline-runs`), showing
    the just-cancelled row's new `"cancelled"` status with its Cancel
    button gone -- the DB write/lock-release/task-cancel behavior itself
    is unchanged (already covered by
    `test_cancel_running_run_returns_200_cancels_row_and_releases_lock`).
    """
    run_id = await _insert_pipeline_run(session_factory, "running")

    async with _make_client(session_factory) as client:
        response = await client.post(
            f"/pipeline/runs/{run_id}/cancel", headers={"HX-Request": "true"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="pipeline-runs">' in response.text
    assert f'hx-post="/pipeline/runs/{run_id}/cancel"' not in response.text

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == "cancelled"


async def test_hx_request_cancel_terminal_run_renders_fragment_not_raw_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#59: the `409` race (the row is already terminal) still responds
    `200 text/html` with the re-rendered fragment when `HX-Request` is
    set -- never a raw/broken error swapped into the page -- while the
    row itself is left completely unmodified (same invariant
    `test_cancel_terminal_run_returns_409_and_does_not_modify_row`
    proves for the non-HTMX JSON path).
    """
    finished_at = datetime.now(UTC)
    run_id = await _insert_pipeline_run(
        session_factory, "ok", finished_at=finished_at, error="pre-existing"
    )

    async with _make_client(session_factory) as client:
        response = await client.post(
            f"/pipeline/runs/{run_id}/cancel", headers={"HX-Request": "true"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="pipeline-runs">' in response.text
    assert f'hx-post="/pipeline/runs/{run_id}/cancel"' not in response.text

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == "ok"
        assert run.error == "pre-existing"
        assert run.finished_at == finished_at


async def test_hx_request_cancel_unknown_run_id_renders_fragment_not_raw_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#59: the `404` race (the id no longer resolves) also responds
    `200 text/html` with the re-rendered fragment when `HX-Request` is
    set, rather than a raw/broken error -- the fragment simply reflects
    the DB's current up-to-5 rows, none of which is the unknown id.
    """
    async with _make_client(session_factory) as client:
        response = await client.post(
            f"/pipeline/runs/{uuid.uuid4()}/cancel", headers={"HX-Request": "true"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<div id="pipeline-runs">' in response.text


async def test_non_htmx_cancel_responses_unchanged_by_hx_request_branch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#59's new `HX-Request` branch must not alter the existing JSON
    contract for a caller that never sends `HX-Request` (e.g. `curl`) --
    success, `409`, and `404` all still return the original JSON bodies,
    exercised together here as a direct regression check alongside the
    pre-existing, unmodified
    `test_cancel_running_run_returns_200_cancels_row_and_releases_lock`/
    `test_cancel_terminal_run_returns_409_and_does_not_modify_row`/
    `test_cancel_unknown_run_id_returns_404` (re-run verbatim, per this
    issue's Constraints).
    """
    running_id = await _insert_pipeline_run(session_factory, "running")
    terminal_id = await _insert_pipeline_run(
        session_factory, "failed", finished_at=datetime.now(UTC), error="boom"
    )

    async with _make_client(session_factory) as client:
        success_response = await client.post(f"/pipeline/runs/{running_id}/cancel")
        assert success_response.status_code == 200
        assert success_response.headers["content-type"].startswith("application/json")
        success_body = success_response.json()
        assert set(success_body.keys()) == _PIPELINE_RUN_SUMMARY_FIELDS
        assert success_body["status"] == "cancelled"

        conflict_response = await client.post(f"/pipeline/runs/{terminal_id}/cancel")
        assert conflict_response.status_code == 409
        assert conflict_response.headers["content-type"].startswith("application/json")
        assert set(conflict_response.json().keys()) == {"detail"}

        missing_response = await client.post(f"/pipeline/runs/{uuid.uuid4()}/cancel")
        assert missing_response.status_code == 404
        assert missing_response.headers["content-type"].startswith("application/json")
        assert set(missing_response.json().keys()) == {"detail"}


# --- #63: auto-refresh the dashboard's pipeline-runs section ---------------
#
# `_render_pipeline_runs_fragment` caps its query at `_RECENT_RUNS_LIMIT`
# (5) most-recent rows (`started_at desc`), and this test DB is shared and
# never truncated across runs. A row inserted `status="running"` by an
# earlier test in this module or `test_web_dashboard.py` that never reaches
# a terminal status (e.g. `test_running_pipeline_run_shows_cancel_button`)
# would otherwise leak into these tests' "top 5" and make "no row is
# running" assertions flaky depending on run history/order. `_clear_
# running_pipeline_runs` forces a known "nothing running" baseline first --
# the same convention `tests/test_scheduler.py`'s `test_lifespan_sweep_is_
# noop_with_zero_running_pipeline_runs` already uses for the identical
# problem.


async def _clear_running_pipeline_runs(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Force every `status="running"` `pipeline_run` row to a terminal
    status, so a #63 polling-attribute test starts from a known baseline
    rather than assuming this shared DB has none already.
    """
    async with session_factory() as session:
        await session.execute(
            update(PipelineRun)
            .where(PipelineRun.status == "running")
            .values(
                status="failed",
                error="test setup: pre-existing stale running row (#63 baseline)",
                finished_at=datetime.now(UTC),
            )
        )
        await session.commit()


async def _finish_pipeline_run(
    session_factory: async_sessionmaker[AsyncSession], run_id: uuid.UUID
) -> None:
    """Force one specific row to a terminal status -- cleanup for a test
    that inserts its own `"running"` row directly (never through `POST
    /pipeline/run`, so nothing else will ever terminate it) so it doesn't
    leak into a later test's "top 5" as a permanent false orphan, same
    `finally`-cleanup convention `tests/test_scheduler.py`'s `test_
    lifespan_sweep_does_not_touch_a_run_started_after_the_sweep` uses.
    """
    async with session_factory() as session:
        await session.execute(
            update(PipelineRun)
            .where(PipelineRun.id == run_id)
            .values(status="ok", finished_at=datetime.now(UTC))
        )
        await session.commit()


async def test_hx_request_list_returns_fragment_with_polling_attributes_when_a_row_is_running(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#63: `GET /pipeline/runs` with `HX-Request: true` returns `200
    text/html`, the re-rendered `partials/pipeline_runs.html` fragment
    (not the JSON array) -- and, since a rendered row has `status ==
    "running"`, the outer `#pipeline-runs` div carries all three polling
    attributes (`hx-get="/pipeline/runs"`,
    `hx-trigger="every 3s [!document.hidden]"` (#64's tab-visibility
    filter), `hx-swap="outerHTML"`), same shape #62's trigger-success
    fragment already has, just reached via a plain `GET` instead of `POST
    /pipeline/run`.
    """
    await _clear_running_pipeline_runs(session_factory)
    run_id = await _insert_pipeline_run(session_factory, "running")

    try:
        async with _make_client(session_factory) as client:
            response = await client.get("/pipeline/runs", headers={"HX-Request": "true"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert (
            '<div id="pipeline-runs" hx-get="/pipeline/runs" '
            'hx-trigger="every 3s [!document.hidden]" hx-swap="outerHTML">' in response.text
        )
        assert f'hx-post="/pipeline/runs/{run_id}/cancel"' in response.text
        assert "Status: <strong>running</strong>" in response.text
    finally:
        await _finish_pipeline_run(session_factory, run_id)


async def test_hx_request_list_omits_polling_attributes_when_every_row_is_terminal(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#63: when every one of the (up to `_RECENT_RUNS_LIMIT`) rendered
    rows is terminal, none of `hx-get`/`hx-trigger`/`hx-swap` are present
    in the rendered HTML at all -- not merely pointing somewhere inert --
    which is what actually stops htmx's polling timer
    (`processPolling`'s `bodyContains(elt)` check never gets a chance to
    reschedule a swapped-in element that was never given a trigger in the
    first place).
    """
    await _clear_running_pipeline_runs(session_factory)
    marker = f"terminal-marker-{uuid.uuid4()}"
    await _insert_pipeline_run(
        session_factory, "ok", finished_at=datetime.now(UTC), error=marker
    )

    async with _make_client(session_factory) as client:
        response = await client.get("/pipeline/runs", headers={"HX-Request": "true"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    # The exact, unadorned opening tag -- proves no polling attributes
    # were added, not just that this particular assertion string is
    # absent.
    assert '<div id="pipeline-runs">' in response.text
    assert "hx-get" not in response.text
    assert "hx-trigger" not in response.text
    assert marker in response.text


async def test_non_htmx_list_response_unchanged_by_hx_request_branch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#63's new `HX-Request` branch must not alter the existing JSON
    contract for a caller that never sends `HX-Request` (e.g. `curl`):
    still the original bare `JSONResponse` array, `_MAX_RUNS`-capped,
    `started_at desc` -- byte-for-byte the same shape
    `test_trigger_run_returns_202_immediately_then_reaches_terminal_status`
    already exercises for this same endpoint, checked here directly and
    in isolation as this issue's own regression test.
    """
    run_id = await _insert_pipeline_run(session_factory, "running")

    try:
        async with _make_client(session_factory) as client:
            response = await client.get("/pipeline/runs")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        body = response.json()
        assert isinstance(body, list)
        assert set(body[0].keys()) == _PIPELINE_RUN_SUMMARY_FIELDS
        ids = [row["id"] for row in body]
        assert str(run_id) in ids
        started_at_values = [row["started_at"] for row in body]
        assert started_at_values == sorted(started_at_values, reverse=True)
    finally:
        await _finish_pipeline_run(session_factory, run_id)


async def test_hx_request_list_poll_reflects_current_db_state_across_transition(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#63: simulates the dashboard's own polling loop hitting `GET
    /pipeline/runs` with `HX-Request` twice -- once while the run is
    still `"running"` (poll response shows `status="running"` and keeps
    the polling attributes present, per this issue's "each poll response
    reflects current DB state" criterion) and once after the row has
    reached a terminal status via a direct DB write standing in for
    `run_pipeline`'s own closing write (poll response shows the terminal
    status and the very next poll omits the polling attributes, per this
    issue's "when the polled run reaches a terminal status" criterion) --
    with no separate caching/staleness layer of its own, straight off
    each request's own DB read.
    """
    await _clear_running_pipeline_runs(session_factory)
    run_id = await _insert_pipeline_run(session_factory, "running")

    async with _make_client(session_factory) as client:
        first_poll = await client.get("/pipeline/runs", headers={"HX-Request": "true"})
        assert first_poll.status_code == 200
        assert "Status: <strong>running</strong>" in first_poll.text
        assert 'hx-trigger="every 3s [!document.hidden]"' in first_poll.text

        async with session_factory() as session:
            await session.execute(
                update(PipelineRun)
                .where(PipelineRun.id == run_id)
                .values(status="ok", finished_at=datetime.now(UTC))
            )
            await session.commit()

        second_poll = await client.get("/pipeline/runs", headers={"HX-Request": "true"})

    assert second_poll.status_code == 200
    assert "Status: <strong>ok</strong>" in second_poll.text
    assert "hx-trigger" not in second_poll.text
    assert '<div id="pipeline-runs">' in second_poll.text


async def test_two_running_rows_keep_polling_attributes_until_both_reach_terminal(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """#63's edge case: with two rows both `status == "running"` at once
    (e.g. a stale row left running from before #53/#60, plus a fresh
    trigger), the polling attributes stay present as long as *at least
    one* is still running -- terminalizing only one of the two must not
    stop polling -- and only disappear once *both* have reached a
    terminal status.
    """
    await _clear_running_pipeline_runs(session_factory)
    first_id = await _insert_pipeline_run(session_factory, "running")
    second_id = await _insert_pipeline_run(session_factory, "running")

    async with _make_client(session_factory) as client:
        both_running = await client.get("/pipeline/runs", headers={"HX-Request": "true"})
        assert 'hx-trigger="every 3s [!document.hidden]"' in both_running.text

        async with session_factory() as session:
            await session.execute(
                update(PipelineRun)
                .where(PipelineRun.id == first_id)
                .values(status="ok", finished_at=datetime.now(UTC))
            )
            await session.commit()

        one_still_running = await client.get(
            "/pipeline/runs", headers={"HX-Request": "true"}
        )
        assert 'hx-trigger="every 3s [!document.hidden]"' in one_still_running.text

        async with session_factory() as session:
            await session.execute(
                update(PipelineRun)
                .where(PipelineRun.id == second_id)
                .values(status="ok", finished_at=datetime.now(UTC))
            )
            await session.commit()

        both_terminal = await client.get("/pipeline/runs", headers={"HX-Request": "true"})

    assert "hx-trigger" not in both_terminal.text
    assert '<div id="pipeline-runs">' in both_terminal.text
