"""Watch status endpoints + page (issue #34; `design.md` §12, §13).

Single-watch MVP: every endpoint here always operates on the Silver watch
(`nie.seed.run.SILVER_WATCH_SLUG`) -- there is no multi-watch selection
anywhere in this module.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import PipelineRun, Watch
from nie.schemas import PipelineRunSummary, WatchStatusResponse
from nie.seed.run import SILVER_WATCH_SLUG
from nie.web.deps import get_session

router = APIRouter()

# `Annotated` form (FastAPI's recommended modern style) rather than a
# `Depends(get_session)` default value -- avoids ruff's B008
# (function-call-in-default-argument) false positive on `Depends`.
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def _get_silver_watch(session: AsyncSession) -> Watch:
    """Fetch the Silver `Watch` row.

    Raises FastAPI's default 404 (a JSON `{"detail": ...}` body, never a
    500) when the DB has been migrated but never seeded, per issue #34's
    acceptance criteria -- this is unconditional on the `HX-Request`
    header, unlike the success-path JSON/HTML branching in `_respond`.
    """
    result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
    watch = result.scalar_one_or_none()
    if watch is None:
        raise HTTPException(status_code=404, detail=f"Watch {SILVER_WATCH_SLUG!r} not found")
    return watch


async def _build_status(session: AsyncSession, watch: Watch) -> WatchStatusResponse:
    """Build the response body: `watch`'s current status plus the most
    recent `pipeline_run` row (by `started_at desc`), or `last_run=None`
    if the pipeline has never run. Not scoped to `watch` -- `PipelineRun`
    (`nie.models`) has no `watch_id` (one run covers the whole pipeline).
    """
    result = await session.execute(
        select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(1)
    )
    run = result.scalar_one_or_none()
    last_run = (
        None
        if run is None
        else PipelineRunSummary(
            id=run.id,
            trigger=run.trigger,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            stats=run.stats,
            error=run.error,
        )
    )
    return WatchStatusResponse(slug=watch.slug, status=watch.status, last_run=last_run)


def _respond(request: Request, status_response: WatchStatusResponse) -> Response:
    """Branch on the incoming `HX-Request` header (set automatically by
    HTMX on every request it makes): an HTMX request gets back the
    rendered `partials/watch_status.html` fragment (`text/html`) so the
    toggle button's swap never triggers a full page reload; a plain
    request gets the JSON `WatchStatusResponse`.
    """
    if request.headers.get("HX-Request"):
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            request, "partials/watch_status.html", {"watch": status_response}
        )
    return JSONResponse(content=status_response.model_dump(mode="json"))


@router.get("/watch", response_class=HTMLResponse)
async def watch_page(request: Request, session: SessionDep) -> HTMLResponse:
    """Render the watch page: current status text and the enable/disable
    toggle button (`design.md` §13).
    """
    watch = await _get_silver_watch(session)
    status_response = await _build_status(session, watch)
    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(request, "watch.html", {"watch": status_response})


@router.get("/watch/status")
async def get_watch_status(request: Request, session: SessionDep) -> Response:
    watch = await _get_silver_watch(session)
    status_response = await _build_status(session, watch)
    return _respond(request, status_response)


@router.post("/watch/enable")
async def enable_watch(request: Request, session: SessionDep) -> Response:
    """Set the Silver watch's status to `"enabled"`. Idempotent: a no-op
    `200` (no write) when it's already enabled.
    """
    watch = await _get_silver_watch(session)
    if watch.status != "enabled":
        watch.status = "enabled"
        await session.commit()
    status_response = await _build_status(session, watch)
    return _respond(request, status_response)


@router.post("/watch/disable")
async def disable_watch(request: Request, session: SessionDep) -> Response:
    """Set the Silver watch's status to `"disabled"`. Idempotent: a no-op
    `200` (no write) when it's already disabled.
    """
    watch = await _get_silver_watch(session)
    if watch.status != "disabled":
        watch.status = "disabled"
        await session.commit()
    status_response = await _build_status(session, watch)
    return _respond(request, status_response)
