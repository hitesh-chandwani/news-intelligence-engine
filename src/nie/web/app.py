"""FastAPI app factory (issue #34, first web-layer code in this repo).

`create_app()` builds a fresh `FastAPI` instance rather than a
module-level singleton so tests can override `nie.web.deps.get_session`
per-app without leaking state between tests, same reasoning
`nie.db.create_engine`/`create_session_factory` use for the DB layer.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from sqlalchemy import update

from nie.db import async_session_factory, engine
from nie.models import PipelineRun
from nie.scheduler import create_scheduler
from nie.web.routers import context, dashboard, events, notifications, pipeline, preferences, watch

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Distinguishes a startup-swept orphan (#60) from a normal stage-raised
# failure in the `error` column at a glance -- see `_reconcile_orphaned_
# pipeline_runs`'s docstring.
ORPHANED_RUN_ERROR = "orphaned: process restarted while this run was still marked running"


async def _reconcile_orphaned_pipeline_runs() -> None:
    """Sweep every `pipeline_run` row still `status="running"` at process
    startup -- these can only be left over from a *previous* process that
    exited without going through `_lifespan`'s graceful-shutdown path
    (`kill -9`, OOM, a crash); see #60.

    Runs before `scheduler.start()` and before any request has been
    served on this process, so `app.state.pipeline_tasks` is guaranteed
    empty at this point -- there is no legitimate in-progress run this
    process could confuse with an orphan, and no age threshold or lock
    force-release is needed (confirmed in #60's grooming). A single bulk
    `UPDATE ... WHERE status = 'running'` (no per-row loop) moves every
    such row to `status="failed"` (reusing the existing terminal status
    and `error` free-text column, same pattern as `pipeline.py`'s
    `CANCELLED_ERROR` -- no new status value, no new migration) with a
    distinct `error` message so it's greppable/distinguishable from a
    normal stage failure in the UI/API, and stamps `finished_at` so it
    stops showing as perpetually in-progress.

    Uses `async_session_factory` directly (`nie.db`) rather than a
    request-scoped `SessionDep` -- `_lifespan` has no request to inject
    one from, same reasoning `nie.pipeline.runner.run_pipeline` uses for
    its own default `session_factory` argument.

    Out of scope (see #60): multiple concurrently-live processes/workers
    sharing one DB -- not this app's deployment model, `design.md`
    §17/§18's single Uvicorn process.

    Disposes `nie.db.engine`'s pool first: it's a process-lifetime
    singleton (module-level, built at import time), and this is the
    first point in a real process's life it's guaranteed to be used, so
    disposing any connection the pool may already be holding (idle from
    an earlier checkout, or -- as in `tests/test_scheduler.py`'s
    lifespan-driving tests, which each run in their own
    `pytest-asyncio`-managed event loop -- bound to a now-closed loop)
    before the very first real checkout is a harmless no-op in
    production and keeps this startup step correct however `_lifespan`
    ends up being invoked.
    """
    await engine.dispose()
    async with async_session_factory() as session:
        await session.execute(
            update(PipelineRun)
            .where(PipelineRun.status == "running")
            .values(status="failed", error=ORPHANED_RUN_ERROR, finished_at=datetime.now(UTC))
        )
        await session.commit()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the `AsyncIOScheduler` (#40) on app startup and shut it down
    cleanly on app shutdown.

    FastAPI's modern `lifespan` context manager, not the deprecated
    `@app.on_event` hook -- per `design.md` §17, Uvicorn starting this one
    process is what "also starts the APScheduler job".

    On startup, before the scheduler starts accepting ticks, first sweeps
    orphaned `pipeline_run` rows (#60) via `_reconcile_orphaned_pipeline_
    runs` -- a previous process that exited via `kill -9`/OOM/a crash
    (rather than this function's own graceful-shutdown path below) can
    leave a row behind still `status="running"` with nothing left to ever
    move it to a terminal status. See that function's own docstring.

    Passes `app.state.pipeline_lock` (set in `create_app()`, not here --
    see that function's docstring) through to `create_scheduler` (#43) so
    the scheduled job and `POST /pipeline/run` share the same
    `asyncio.Lock` instance and never run `run_pipeline` concurrently.

    On shutdown (#54), also cancels every `asyncio.Task` still tracked in
    `app.state.pipeline_tasks` -- a background run spawned by
    `POST /pipeline/run` that is still in flight when the app shuts down
    gracefully must not be silently dropped or orphaned. Tasks are
    cancelled and then awaited (`asyncio.gather(..., return_exceptions=
    True)`) so shutdown doesn't proceed while a task is still mid-cleanup,
    but each task's own background wrapper is responsible for releasing
    `pipeline_lock` and popping its own `pipeline_tasks` entry in a
    `finally`, so this never hangs waiting on real pipeline I/O -- only
    on each task noticing the cancellation at its next `await` point.
    """
    await _reconcile_orphaned_pipeline_runs()
    scheduler = create_scheduler(lock=app.state.pipeline_lock)
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        tasks: dict[uuid.UUID, asyncio.Task[None]] = app.state.pipeline_tasks
        if tasks:
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)


def create_app() -> FastAPI:
    """Build the FastAPI app: `Jinja2Templates` on `app.state.templates`
    (read by the routers to render `templates/`), the watch (#34),
    context (#35), preferences (#36), events (#37), notifications (#38),
    pipeline (#43), and dashboard (#49) routers included -- `design.md`
    §12's full router table is added incrementally by later tasks -- plus
    the `AsyncIOScheduler` (#40) started/stopped via `lifespan`.

    `app.state.pipeline_lock` (issue #43) is set directly here, not
    inside `_lifespan`: `httpx.ASGITransport` (used by every existing web
    test except `test_scheduler.py`'s one lifespan-driving test) never
    fires ASGI lifespan events, so a lock created only inside `_lifespan`
    would leave `request.app.state.pipeline_lock` unset -- an
    `AttributeError` -- for every other router's tests, and for
    `POST /pipeline/run`/`GET /pipeline/runs` themselves whenever they're
    exercised the same way. Creating exactly one `asyncio.Lock()` here
    keeps it a true process-local singleton, shared by the scheduler
    (via `_lifespan`, above) and the pipeline router.

    `app.state.pipeline_tasks: dict[uuid.UUID, asyncio.Task]` (#54) is
    set here for the same reason: a plain dict, empty at startup, keyed
    by `PipelineRun.id`, holding a live reference to every in-flight
    background run `POST /pipeline/run` spawns via
    `asyncio.create_task()` -- `asyncio` itself only keeps a weak
    reference to a created task, so an otherwise-unreferenced one risks
    being garbage-collected mid-run. `trigger_pipeline_run`
    (`src/nie/web/routers/pipeline.py`) stores the task here immediately
    after creating it, and the background wrapper it spawns pops its own
    entry in a `finally` block, so this dict always reflects exactly
    which runs currently have a live task backing them.
    """
    app = FastAPI(title="News Intelligence Engine", lifespan=_lifespan)
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.pipeline_lock = asyncio.Lock()
    app.state.pipeline_tasks = {}
    app.include_router(watch.router)
    app.include_router(context.router)
    app.include_router(preferences.router)
    app.include_router(events.router)
    app.include_router(notifications.router)
    app.include_router(pipeline.router)
    app.include_router(dashboard.router)
    return app
