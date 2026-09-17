"""Manual pipeline trigger + run history API (issue #43; `design.md` §12).

`POST /pipeline/run` and `GET /pipeline/runs` are the two endpoints
`design.md` §12's HTTP API table lists for this router. Unlike every
other router in `nie.web.routers`, neither endpoint is scoped to the
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
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
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


@router.post("/pipeline/run")
async def trigger_pipeline_run(request: Request, session: SessionDep) -> JSONResponse:
    """Trigger one pipeline run right now, with `trigger="manual"`.

    If `app.state.pipeline_lock` is already held (a previous manual
    trigger or a scheduled tick is still inside `run_pipeline`), responds
    `409` immediately -- `lock.locked()` is checked before attempting
    acquisition so this never blocks waiting for the lock to free up.

    Otherwise, acquires the lock, awaits `run_pipeline(...)` for the
    *entire* run (this can legitimately take minutes -- there is no task
    queue in `design.md`'s MVP stack; #54 tracks replacing this with
    background execution + polling), releases the lock, and returns `200`
    with the finished run as a `PipelineRunSummary`. `status` in the
    response is always a terminal value (`"ok"`/`"partial"`/`"failed"`),
    never `"running"`, because `run_pipeline` doesn't return until the
    run has finished.

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

    async with lock:
        run_id = await run_pipeline(trigger=MANUAL_TRIGGER)

    result = await session.execute(select(PipelineRun).where(PipelineRun.id == run_id))
    run = result.scalar_one()
    return JSONResponse(content=_to_summary(run).model_dump(mode="json"))


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
