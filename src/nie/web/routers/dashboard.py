"""Dashboard page (issue #49; `design.md` §12, §13).

`GET /` is the app's landing page: it assembles the Silver watch's current
status + toggle (#34), the recent pipeline runs section (originally a
single most-recent-run summary from #34, replaced by an up-to-5-rows
"Recent pipeline runs" section with a Cancel button in #59), the
most-recent events (#37), and the unread-notification count (#38) into one
read-only page, built mostly from query logic those issues already
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

# `GET /`'s "Recent pipeline runs" section (#59) shows at most this many
# rows, most-recent-first -- same small-fixed-cap, no-filter-controls
# shape `_RECENT_EVENTS_LIMIT` already established for Recent events.
_RECENT_RUNS_LIMIT = 5


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


async def _recent_pipeline_runs(session: AsyncSession) -> list[PipelineRunSummary]:
    """The `_RECENT_RUNS_LIMIT` most recent `pipeline_run` rows
    (`started_at desc`), or an empty list if the pipeline has never run
    (#59; replaces the old single-row `_most_recent_pipeline_run`).

    Same query `GET /pipeline/runs` (`src/nie/web/routers/pipeline.py`)
    already runs, capped the same "reuse the query, slice/limit in
    Python" way `_RECENT_EVENTS_LIMIT` already caps Recent events --
    duplicated here rather than cross-imported (this module defines its
    own small "recent pipeline_run rows" query per the issue's
    Constraints, rather than importing `pipeline.py`'s query logic). Not
    scoped to a watch -- `PipelineRun` (`nie.models`) has no `watch_id`
    (one run covers the whole pipeline).
    """
    result = await session.execute(
        select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(_RECENT_RUNS_LIMIT)
    )
    runs = result.scalars().all()
    return [
        PipelineRunSummary(
            id=run.id,
            trigger=run.trigger,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            stats=run.stats,
            error=run.error,
        )
        for run in runs
    ]


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request, session: SessionDep) -> HTMLResponse:
    """Render the dashboard: watch status + toggle, recent pipeline runs
    (capped at `_RECENT_RUNS_LIMIT`, #59), recent events (capped at
    `_RECENT_EVENTS_LIMIT`), and the unread-notification count, each
    with links to their own full page (`/events`, `/notifications`,
    `/context`, `/preferences`).

    Recent events reuse `events.py`'s `_list_events` with every filter
    `None` (unfiltered, `event_date desc` nulls last, tie-broken
    `discovered_at desc` -- the ordering `events.py` already established),
    then sliced to the small display cap in Python, per the issue's
    Constraints. The unread count is computed in Python from
    `nie.notifications.inbox.list_notifications`'s existing result
    (counting `read_at is None`) -- no new query function added to
    `inbox.py`.

    The pipeline-runs section is rendered via the shared
    `partials/pipeline_runs.html` fragment (#59) -- the same fragment
    `POST /pipeline/runs/{run_id}/cancel`'s `HX-Request` branch
    (`pipeline.py`) re-renders after a cancel, so a running row's
    Cancel button always swaps into an up-to-date view of this same
    section.
    """
    watch = await _get_silver_watch(session)
    pipeline_runs = await _recent_pipeline_runs(session)
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
            "pipeline_runs": pipeline_runs,
            "events": events,
            "unread_count": unread_count,
        },
    )
