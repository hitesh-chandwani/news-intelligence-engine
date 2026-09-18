"""Manual pipeline trigger + run history API (issue #43, #53, #54;
`design.md` §12).

`POST /pipeline/run`, `GET /pipeline/runs`, `GET /pipeline/runs/{run_id}`
(#54), and `POST /pipeline/runs/{run_id}/cancel` (#53) are the four
endpoints `design.md` §12's HTTP API table lists for this router. Unlike
every other router in `nie.web.routers`, none of them are scoped to the
Silver watch (`nie.seed.run.SILVER_WATCH_SLUG`): `PipelineRun`
(`src/nie/models.py`) has no `watch_id` column at all -- one run covers
the whole (single-watch, for MVP) pipeline -- and `POST /pipeline/run`
deliberately does not look up or gate on `Watch.status` either (see that
route's docstring for why).

Overlap protection with the scheduler (#40) is `request.app.state.
pipeline_lock`, the single process-local `asyncio.Lock` `create_app()`
(`src/nie/web/app.py`) creates and `_lifespan` also hands to
`create_scheduler()` -- see that module's docstrings for the full
reasoning (a process-local lock over APScheduler's own `max_instances`,
which is enforced per job id, not globally).

`POST /pipeline/run` (#54) no longer awaits `run_pipeline()` for the
whole run. It synchronously acquires `pipeline_lock`, inserts the
`pipeline_run` row itself, and spawns the run as a tracked
`asyncio.Task` (`app.state.pipeline_tasks`) *before* returning `202` --
see `trigger_pipeline_run`'s own docstring for the exact sequencing and
why it matters. `POST /pipeline/runs/{run_id}/cancel` (#53) is upgraded
accordingly: on a successful DB cancel it now also calls `.cancel()` on
that tracked task, giving real interruption rather than a DB-only flip
-- see `cancel_pipeline_run`'s docstring for the one known limitation
(`embed_stage`'s blocking, non-executor-wrapped call). Both the cancel
endpoint's row write and `run_pipeline`'s own closing write remain
atomic conditional `UPDATE ... WHERE status = 'running'` statements (see
each one's docstring) specifically so those two competing writers can
never clobber each other, independent of whether the cancel also landed
a `task.cancel()`.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import PipelineRun
from nie.pipeline.runner import run_pipeline
from nie.schemas import PipelineRunSummary
from nie.web.deps import get_session

router = APIRouter()

# `Annotated` form (FastAPI's recommended modern style) rather than a
# `Depends(get_session)` default value -- avoids ruff's B008
# (function-call-in-default-argument) false positive on `Depends`, same
# pattern every other router in this package uses.
SessionDep = Annotated[AsyncSession, Depends(get_session)]

# The literal trigger value `run_pipeline` is called with here. Must be
# exactly "manual" (not e.g. "manual_trigger") -- `PipelineRun`'s DB
# `CheckConstraint ck_pipeline_run_trigger`
# (`alembic/versions/a0023c0d85d2_pipeline_run_table.py`) only allows
# 'schedule'/'manual'; any other value fails the insert.
MANUAL_TRIGGER = "manual"

# `GET /pipeline/runs` returns at most this many rows, most-recent-first
# -- no cursor/offset pagination, same precedent `events.py`'s
# `_MAX_EVENTS` set for `GET /events` (#37).
_MAX_RUNS = 100

# `error` value `POST /pipeline/runs/{run_id}/cancel` writes on a
# successful cancel (#53) -- distinguishes an operator cancel from a
# stage-raised `error` string in the `pipeline_run.error` column.
CANCELLED_ERROR = "cancelled by operator"


def _to_summary(run: PipelineRun) -> PipelineRunSummary:
    return PipelineRunSummary(
        id=run.id,
        trigger=run.trigger,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        stats=run.stats,
        error=run.error,
    )


async def _run_pipeline_in_background(app: FastAPI, run_id: uuid.UUID) -> None:
    """Background wrapper `trigger_pipeline_run` spawns via
    `asyncio.create_task()` -- the actual `run_pipeline()` call for a
    manual trigger, running on the same event loop after the request
    handler has already returned `202` (#54).

    Owns releasing `app.state.pipeline_lock` and removing this run's
    entry from `app.state.pipeline_tasks`, both in a `finally` so they
    happen exactly once no matter how the run ends: success, a stage
    exception (already caught and recorded by `run_pipeline` itself,
    never raised out of it), or cancellation (`asyncio.CancelledError`,
    raised by `.cancel()` on this task -- see
    `cancel_pipeline_run`). The lock release is guarded by
    `if lock.locked()` because a successful cancel via
    `cancel_pipeline_run` already force-released it (it does not wait for
    this wrapper to notice); this is a no-op in that case, not a double
    release. `asyncio.CancelledError` is caught around the `run_pipeline`
    call itself (not silently left to propagate out of this task) so an
    externally cancelled run doesn't surface as an unhandled-task-
    exception log -- the cancel was deliberate, initiated by
    `cancel_pipeline_run`, and already fully accounted for in the DB by
    the time `.cancel()` is called.
    """
    lock: asyncio.Lock = app.state.pipeline_lock
    try:
        await run_pipeline(trigger=MANUAL_TRIGGER, run_id=run_id)
    except asyncio.CancelledError:
        pass
    finally:
        if lock.locked():
            lock.release()
        app.state.pipeline_tasks.pop(run_id, None)


@router.post("/pipeline/run", status_code=202)
async def trigger_pipeline_run(request: Request, session: SessionDep) -> JSONResponse:
    """Trigger one pipeline run right now, with `trigger="manual"`,
    returning `202 Accepted` immediately rather than blocking for the
    full run (#54).

    If `app.state.pipeline_lock` is already held (a previous manual
    trigger or a scheduled tick is still inside `run_pipeline`), responds
    `409` immediately -- `lock.locked()` is checked before attempting
    acquisition so this never blocks waiting for the lock to free up.

    Otherwise, the following happens synchronously, in this order,
    *before* returning `202` -- see `design.md` §12/issue #54's
    "Lock-acquisition race" for why this exact sequence matters:

    1. `await lock.acquire()`. On the uncontended fast path (true here,
       since `lock.locked()` above was just checked `False`)
       `asyncio.Lock.acquire()` returns without an actual suspension
       point, so no other coroutine -- including another concurrent
       `POST /pipeline/run` -- can run between the `locked()` check above
       and this acquisition. This is the same invariant #43's original
       `if lock.locked(): ... async with lock:` already relied on, now
       load-bearing for #54 too: a second request cannot slip in and see
       the lock still free.
    2. Still holding the lock, this handler -- not the background task --
       inserts the `PipelineRun(trigger="manual", status="running",
       stats={})` row itself and commits it, capturing `run_id`. A second
       concurrent request is already rejected at the `lock.locked()`
       check by this point, so there is no separate check-and-create race
       to solve for row creation.
    3. Spawns `_run_pipeline_in_background(request.app, run_id)` as a
       tracked `asyncio.Task` and immediately stores it in
       `request.app.state.pipeline_tasks[run_id]` -- `asyncio` only holds
       a weak reference to a task once created, so storing it here is
       what keeps it alive for the run's duration.

    Only then does this return `202` with the freshly inserted row as a
    `PipelineRunSummary` (`status="running"`, `finished_at=None`). The
    run itself executes on this same event loop, still serialized end to
    end by `pipeline_lock` (released only when the background wrapper's
    `finally` runs, on completion or cancellation) -- a caller polls
    `GET /pipeline/runs/{run_id}` for the run reaching a terminal status.
    A client disconnecting immediately after receiving the `202` has no
    effect on the run: the task is not bound to the request's lifecycle
    at all, only to `request.app.state`.

    Deliberately does not look up or gate on `Watch.status` at all,
    unlike `nie.scheduler.scheduler_tick` -- a manual trigger is an
    explicit "run it now" request from someone actively waiting on this
    response; silently no-op'ing it because the watch happens to be
    disabled would be worse UX than the scheduler's silent skip of an
    unattended interval firing that nobody is watching.
    """
    lock: asyncio.Lock = request.app.state.pipeline_lock

    if lock.locked():
        return JSONResponse(
            status_code=409, content={"detail": "a pipeline run is already in progress"}
        )

    await lock.acquire()

    run = PipelineRun(trigger=MANUAL_TRIGGER, status="running", stats={})
    session.add(run)
    await session.commit()
    await session.refresh(run)
    run_id = run.id

    task = asyncio.create_task(_run_pipeline_in_background(request.app, run_id))
    request.app.state.pipeline_tasks[run_id] = task

    return JSONResponse(status_code=202, content=_to_summary(run).model_dump(mode="json"))


@router.post("/pipeline/runs/{run_id}/cancel")
async def cancel_pipeline_run(
    request: Request, run_id: uuid.UUID, session: SessionDep
) -> JSONResponse:
    """Mark a `status="running"` `pipeline_run` row `"cancelled"`,
    forcibly release `app.state.pipeline_lock`, and (#54) really
    interrupt the run's tracked `asyncio.Task` if one is present.

    The DB write is a single `UPDATE ... WHERE id = :id AND
    status = 'running'` (via SQLAlchemy's `update()`, not a `SELECT`
    followed by a separate `UPDATE`) so a run that finishes naturally in
    the gap between an operator's decision to cancel and this request
    landing is never incorrectly cancelled -- `result.rowcount` is the
    single source of truth for whether this request actually won the
    race, not a preceding read.

    - `rowcount == 1`: the row was `"running"` and is now `"cancelled"`.
      Releases `pipeline_lock` (guarded by `if lock.locked()`, since
      `asyncio.Lock.release()` has no owner check and raises on an
      already-unlocked lock -- this release is forced regardless of
      which coroutine originally acquired it), looks up
      `request.app.state.pipeline_tasks.get(run_id)` and, if present,
      calls `.cancel()` on it (#54's real interruption -- see below), and
      returns `200` with the updated row as a `PipelineRunSummary`. A run
      whose row is `"running"` but has no tracked task (tracking lost
      some other way, or it simply predates #54) still succeeds here as a
      DB-only cancel exactly like before -- `.get(run_id)` returning
      `None` is not an error.
    - `rowcount == 0` and the id exists: the row was already terminal
      (`"ok"`, `"partial"`, `"failed"`, or already `"cancelled"`) --
      returns `409` with a body explaining the run isn't running. Not a
      silent no-op.
    - `rowcount == 0` and the id doesn't exist at all: returns `404`.
      Distinguishing this from the `409` case costs one extra `SELECT`,
      run only after the atomic write already missed -- it plays no part
      in the actual cancel decision, only in choosing which error to
      report.

    **Real interruption, and its one known limitation.** `task.cancel()`
    raises `asyncio.CancelledError` inside the task's coroutine at its
    next `await` point. `run_pipeline`'s stage loop (`runner.py`) only
    catches `except Exception`, which does **not** catch
    `asyncio.CancelledError` (a `BaseException` subclass since Python
    3.8) -- so the cancellation propagates out of whichever stage is
    currently awaited, out of `run_pipeline` entirely, and into
    `_run_pipeline_in_background`'s own `try/except
    asyncio.CancelledError`, whose `finally` releases `pipeline_lock`
    (already force-released above, so a no-op there) and pops the
    `pipeline_tasks` entry. Because cancellation only takes effect at an
    `await`, and every stage but one is I/O-bound (LLM calls, DB
    queries) and yields promptly, this is effectively immediate for
    those stages. The one exception: `embed_stage`
    (`src/nie/pipeline/embed.py`) calls
    `nie.embeddings.fastembed.embed_texts` directly -- a synchronous,
    CPU-bound ONNX call not wrapped in `asyncio.to_thread`/an executor --
    so a cancel requested while embedding is in flight will not interrupt
    it until that call returns on its own. This is a pre-existing
    limitation of `embed_stage` (it already blocks the whole event loop
    during embedding, independent of cancellation), not something fixed
    by this endpoint.
    """
    cancel_result = await session.execute(
        update(PipelineRun)
        .where(PipelineRun.id == run_id, PipelineRun.status == "running")
        .values(status="cancelled", finished_at=datetime.now(UTC), error=CANCELLED_ERROR)
    )
    await session.commit()

    # `AsyncSession.execute()` is typed as returning `Result[Any]`, which
    # has no `.rowcount` -- but executing a Core `update()` statement
    # actually returns a `CursorResult` at runtime (it wraps the DBAPI
    # cursor), which does. Same "the stub is stricter than the runtime
    # type" situation as the `# type: ignore[arg-type]` uses elsewhere in
    # this codebase (e.g. `events.py`, `notify.py`).
    if cancel_result.rowcount == 1:  # type: ignore[attr-defined]
        lock: asyncio.Lock = request.app.state.pipeline_lock
        if lock.locked():
            lock.release()
        task = request.app.state.pipeline_tasks.get(run_id)
        if task is not None:
            task.cancel()
        result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
        run = result.scalar_one()
        return JSONResponse(content=_to_summary(run).model_dump(mode="json"))

    result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
    existing_run = result.scalar_one_or_none()
    if existing_run is None:
        return JSONResponse(status_code=404, content={"detail": f"pipeline run {run_id} not found"})
    detail = f"pipeline run {run_id} is not running (status={existing_run.status!r})"
    return JSONResponse(status_code=409, content={"detail": detail})


@router.get("/pipeline/runs")
async def list_pipeline_runs(session: SessionDep) -> JSONResponse:
    """Every `pipeline_run` row, most recent first (`started_at desc`),
    capped at `_MAX_RUNS` -- a bare JSON array (`JSONResponse(content=
    [...])`, not an object wrapper), same convention `GET /events`/
    `GET /notifications` use. Never scoped to a watch: `PipelineRun` has
    no `watch_id` column.

    A run still `status="running"` (if any -- e.g. `POST /pipeline/run`
    is mid-flight in another request) is included like any other row,
    with `finished_at`/`error` simply `None`.
    """
    result = await session.execute(
        select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(_MAX_RUNS)
    )
    runs = result.scalars().all()
    return JSONResponse(content=[_to_summary(run).model_dump(mode="json") for run in runs])


@router.get("/pipeline/runs/{run_id}")
async def get_pipeline_run(run_id: uuid.UUID, session: SessionDep) -> JSONResponse:
    """One `pipeline_run` row by id, as a `PipelineRunSummary` (#54).

    `200` for any known id regardless of its current `status`, including
    `"running"` -- this is the endpoint a caller polls after `POST
    /pipeline/run`'s `202` to observe the status transition to a
    terminal value. `404` for an unknown id.

    Added alongside `GET /pipeline/runs` rather than only relying on
    polling the list: the list is capped at `_MAX_RUNS` and returns every
    row on every poll for information about exactly one, needs
    client-side search-by-id, and (unlike the list) gives a clean `404`
    for an unknown id -- matching the precedent `POST /pipeline/runs/
    {run_id}/cancel` already set, and mirroring `design.md` §12's
    existing `GET /events` + `GET /events/{id}` pairing. Implementation
    mirrors `cancel_pipeline_run`'s own `SELECT ...
    scalar_one_or_none()` 404 path.
    """
    result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        return JSONResponse(status_code=404, content={"detail": f"pipeline run {run_id} not found"})
    return JSONResponse(content=_to_summary(run).model_dump(mode="json"))
