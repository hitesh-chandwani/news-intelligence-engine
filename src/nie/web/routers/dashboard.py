"""Dashboard page (issue #49; `design.md` §12, §13).

`GET /` is the app's landing page: it assembles the Silver watch's current
status + toggle (#34), a summary of the most recent pipeline run (#34), the
most-recent events (#37), and the unread-notification count (#38) into one
read-only page, built entirely from query logic those three issues already
shipped -- no new domain/business logic here.

New file rather than an extension of `watch.py`: `GET /` aggregates four
independent domains (watch, pipeline_run, event, notification), while every
other router in this app owns exactly one URL prefix/domain
(`watch.py`->`/watch`, `events.py`->`/events`, `notifications.py`->
`/notifications`, `pipeline.py`->`/pipeline`). A new thin router -- the same
"small, aggregating, no big domain logic of its own" shape `pipeline.py`
already has -- is the better fit here, matching this issue's Constraints.

`GET /` is HTML-only: no `HX-Request`/`Accept` JSON branching, unlike every
other page route in this app. `design.md` §12 lists `GET /` once, as the
page itself, not split into a page + a separate JSON/HTMX endpoint. The only
interactive control on this page (the watch toggle, via #34's existing
`partials/watch_status.html`) already gets its own HTMX fragment from
`/watch/enable`/`/watch/disable` regardless of which page embeds it -- this
page adds no write endpoint and no `<form>` of its own, so neither #35's
`hx-ext="json-enc"` requirement nor #36's checkbox-group-serialization bug
class applies here.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import PipelineRun, Watch
from nie.notifications.inbox import list_notifications
from nie.schemas import PipelineRunSummary
from nie.seed.run import SILVER_WATCH_SLUG
from nie.web.deps import get_session
from nie.web.routers.events import _list_events

router = APIRouter()

# Same `Annotated` form every other router in this app uses (FastAPI's
# recommended modern style) -- avoids ruff's B008 (function-call-in-
# default-argument) false positive on `Depends`.
SessionDep = Annotated[AsyncSession, Depends(get_session)]

# `GET /`'s recent-events section shows at most this many rows,
# most-recent-first -- a small fixed cap, no filter controls of its own
# (filtering is the full Timeline page, #37, linked from here).
_RECENT_EVENTS_LIMIT = 5


async def _get_silver_watch(session: AsyncSession) -> Watch:
    """Fetch the Silver `Watch` row, or FastAPI's default `404` (a JSON
    `{"detail": ...}` body) when the DB has been migrated but never seeded
    -- identical to `watch.py`/`context.py`/`preferences.py`/`events.py`/
    `notifications.py`'s `_get_silver_watch`, duplicated here rather than
    cross-imported per this codebase's "duplicate small private helpers"
    convention.
    """
    result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
    watch = result.scalar_one_or_none()
    if watch is None:
        raise HTTPException(status_code=404, detail=f"Watch {SILVER_WATCH_SLUG!r} not found")
    return watch


async def _most_recent_pipeline_run(session: AsyncSession) -> PipelineRunSummary | None:
    """The most recent `pipeline_run` row (`started_at desc`), or `None`
    if the pipeline has never run -- the same few-line query `watch.py`'s
    `_build_status` already computes, duplicated here rather than
    cross-imported (this module defines its own small "most recent
    pipeline_run" query per the issue's Constraints, rather than importing
    `watch.py`'s private `_build_status`). Not scoped to a watch --
    `PipelineRun` (`nie.models`) has no `watch_id` (one run covers the
    whole pipeline).
    """
    result = await session.execute(
        select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(1)
    )
    run = result.scalar_one_or_none()
    if run is None:
        return None
    return PipelineRunSummary(
        id=run.id,
        trigger=run.trigger,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        stats=run.stats,
        error=run.error,
    )


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request, session: SessionDep) -> HTMLResponse:
    """Render the dashboard: watch status + toggle, last-run summary,
    recent events (capped at `_RECENT_EVENTS_LIMIT`), and the unread-
    notification count, each with links to their own full page
    (`/events`, `/notifications`, `/context`, `/preferences`).

    Recent events reuse `events.py`'s `_list_events` with every filter
    `None` (unfiltered, `event_date desc` nulls last, tie-broken
    `discovered_at desc` -- the ordering `events.py` already established),
    then sliced to the small display cap in Python, per the issue's
    Constraints. The unread count is computed in Python from
    `nie.notifications.inbox.list_notifications`'s existing result
    (counting `read_at is None`) -- no new query function added to
    `inbox.py`.
    """
    watch = await _get_silver_watch(session)
    last_run = await _most_recent_pipeline_run(session)
    events = (
        await _list_events(session, watch, None, None, None, None, None)
    )[:_RECENT_EVENTS_LIMIT]
    notifications = await list_notifications(session, watch.id)
    unread_count = sum(1 for notification in notifications if notification.read_at is None)

    templates: Jinja2Templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "watch": watch,
            "last_run": last_run,
            "events": events,
            "unread_count": unread_count,
        },
    )
