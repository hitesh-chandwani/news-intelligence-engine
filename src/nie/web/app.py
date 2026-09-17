"""FastAPI app factory (issue #34, first web-layer code in this repo).

`create_app()` builds a fresh `FastAPI` instance rather than a
module-level singleton so tests can override `nie.web.deps.get_session`
per-app without leaking state between tests, same reasoning
`nie.db.create_engine`/`create_session_factory` use for the DB layer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from nie.scheduler import create_scheduler
from nie.web.routers import context, events, notifications, preferences, watch

TEMPLATES_DIR = Path(__file__).parent / "templates"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the `AsyncIOScheduler` (#40) on app startup and shut it down
    cleanly on app shutdown.

    FastAPI's modern `lifespan` context manager, not the deprecated
    `@app.on_event` hook -- per `design.md` §17, Uvicorn starting this one
    process is what "also starts the APScheduler job".
    """
    scheduler = create_scheduler()
    scheduler.start()
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


def create_app() -> FastAPI:
    """Build the FastAPI app: `Jinja2Templates` on `app.state.templates`
    (read by the routers to render `templates/`), the watch (#34),
    context (#35), preferences (#36), events (#37), and notifications
    (#38) routers included -- `design.md` §12's full router table is added
    incrementally by later tasks -- plus the `AsyncIOScheduler` (#40)
    started/stopped via `lifespan`.
    """
    app = FastAPI(title="News Intelligence Engine", lifespan=_lifespan)
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.include_router(watch.router)
    app.include_router(context.router)
    app.include_router(preferences.router)
    app.include_router(events.router)
    app.include_router(notifications.router)
    return app
