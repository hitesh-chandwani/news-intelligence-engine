"""FastAPI app factory (issue #34, first web-layer code in this repo).

`create_app()` builds a fresh `FastAPI` instance rather than a
module-level singleton so tests can override `nie.web.deps.get_session`
per-app without leaking state between tests, same reasoning
`nie.db.create_engine`/`create_session_factory` use for the DB layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from nie.scheduler import create_scheduler
from nie.web.routers import context, dashboard, events, notifications, pipeline, preferences, watch

TEMPLATES_DIR = Path(__file__).parent / "templates"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the `AsyncIOScheduler` (#40) on app startup and shut it down
    cleanly on app shutdown.

    FastAPI's modern `lifespan` context manager, not the deprecated
    `@app.on_event` hook -- per `design.md` §17, Uvicorn starting this one
    process is what "also starts the APScheduler job".

    Passes `app.state.pipeline_lock` (set in `create_app()`, not here --
    see that function's docstring) through to `create_scheduler` (#43) so
    the scheduled job and `POST /pipeline/run` share the same
    `asyncio.Lock` instance and never run `run_pipeline` concurrently.
    """
    scheduler = create_scheduler(lock=app.state.pipeline_lock)
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


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
    """
    app = FastAPI(title="News Intelligence Engine", lifespan=_lifespan)
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.pipeline_lock = asyncio.Lock()
    app.include_router(watch.router)
    app.include_router(context.router)
    app.include_router(preferences.router)
    app.include_router(events.router)
    app.include_router(notifications.router)
    app.include_router(pipeline.router)
    app.include_router(dashboard.router)
    return app
