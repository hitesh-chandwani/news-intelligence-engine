"""FastAPI app factory (issue #34, first web-layer code in this repo).

`create_app()` builds a fresh `FastAPI` instance rather than a
module-level singleton so tests can override `nie.web.deps.get_session`
per-app without leaking state between tests, same reasoning
`nie.db.create_engine`/`create_session_factory` use for the DB layer.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from nie.web.routers import context, preferences, watch

TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app() -> FastAPI:
    """Build the FastAPI app: `Jinja2Templates` on `app.state.templates`
    (read by the routers to render `templates/`), the watch (#34),
    context (#35), and preferences (#36) routers included -- `design.md`
    §12's full router table is added incrementally by later tasks.
    """
    app = FastAPI(title="News Intelligence Engine")
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.include_router(watch.router)
    app.include_router(context.router)
    app.include_router(preferences.router)
    return app
