"""Notification inbox endpoints + page (issue #38; `design.md` §12, §13).

Single-watch MVP: `GET /notifications` always scopes to the Silver watch
(`nie.seed.run.SILVER_WATCH_SLUG`) -- same precedent `watch.py`/
`context.py`/`preferences.py`/`events.py` established. `POST
/notifications/{id}/read` is not scoped to the Silver watch in its lookup
(an id is already globally unique), same as `events.py`'s `GET
/events/{id}`.

Built entirely on #33's already-shipped data layer
(`nie.notifications.inbox.list_notifications`/`mark_read`) -- this module
adds the HTTP/HTML layer only, calling those two functions directly with
no duplicated query logic. `POST /notifications/{id}/read` takes no
request body (modeled on `POST /watch/enable`/`/watch/disable`'s bodyless
`<button hx-post="...">` pattern, not a `<form>`), so neither #35's
`hx-ext="json-enc"` bug class nor #36's checkbox-serialization bug class
applies here.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import Notification, Watch
from nie.notifications.inbox import list_notifications, mark_read
from nie.schemas import NotificationResponse
from nie.seed.run import SILVER_WATCH_SLUG
from nie.web.deps import get_session

router = APIRouter()

# Same `Annotated` form `watch.py`/`context.py`/`preferences.py`/
# `events.py` use (FastAPI's recommended modern style) -- avoids ruff's
# B008 (function-call-in-default-argument) false positive on `Depends`.
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def _get_silver_watch(session: AsyncSession) -> Watch:
    """Fetch the Silver `Watch` row, or FastAPI's default `404` (a JSON
    `{"detail": ...}` body) when the DB has been migrated but never
    seeded -- identical to `watch.py`/`context.py`/`preferences.py`/
    `events.py`'s `_get_silver_watch`.
    """
    result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
    watch = result.scalar_one_or_none()
    if watch is None:
        raise HTTPException(status_code=404, detail=f"Watch {SILVER_WATCH_SLUG!r} not found")
    return watch


def _to_response(notification: Notification) -> NotificationResponse:
    return NotificationResponse(
        id=notification.id,
        watch_id=notification.watch_id,
        event_id=notification.event_id,
        reason=notification.reason,  # type: ignore[arg-type]
        payload=notification.payload,
        channels_sent=list(notification.channels_sent),
        created_at=notification.created_at,
        read_at=notification.read_at,
    )


@router.get("/notifications")
async def list_notifications_route(request: Request, session: SessionDep) -> Response:
    """Three-way negotiation, same pattern `events.py`'s `list_events`
    (#37) established:

    - `HX-Request: true` -> `200 text/html`, the `partials/
      notification_list.html` fragment (list only)
    - no `HX-Request` and `Accept` starts with `text/html` (a plain
      browser navigation) -> `200 text/html`, the full page
      (`notifications.html`, list fragment + page chrome)
    - anything else (including `httpx`'s default `*/*`) -> `200
      application/json`, a JSON array of `NotificationResponse`

    Always scoped to the Silver watch via `list_notifications(session,
    watch.id)`; `404` if the watch doesn't exist, same as every other
    router in this module.
    """
    watch = await _get_silver_watch(session)
    notifications = await list_notifications(session, watch.id)
    responses = [_to_response(notification) for notification in notifications]

    templates: Jinja2Templates = request.app.state.templates
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request, "partials/notification_list.html", {"notifications": responses}
        )

    accept = request.headers.get("accept", "")
    if accept.startswith("text/html"):
        return templates.TemplateResponse(
            request, "notifications.html", {"notifications": responses}
        )

    return JSONResponse(content=[response.model_dump(mode="json") for response in responses])


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(
    request: Request, notification_id: uuid.UUID, session: SessionDep
) -> Response:
    """Mark one notification read, then respond per the `HX-Request`
    header. Takes no request body -- the id is already in the URL path
    and there is nothing else to submit (see the issue's Constraints).

    Calls `mark_read(session, notification_id)`; `mark_read` itself is
    HTTP-agnostic per #33 and raises a plain `ValueError` for an unknown
    id, so that translation to `HTTPException(404)` (JSON
    `{"detail": ...}`) happens here in the router.

    On success: `HX-Request` -> the re-rendered `partials/
    notification_list.html` fragment for this notification's watch,
    reflecting the now-read state (satisfies "the list reflects read
    state without a full reload"); otherwise -> JSON
    `NotificationResponse` with `read_at` set.
    """
    try:
        notification = await mark_read(session, notification_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if request.headers.get("HX-Request"):
        notifications = await list_notifications(session, notification.watch_id)
        responses = [_to_response(item) for item in notifications]
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            request, "partials/notification_list.html", {"notifications": responses}
        )

    return JSONResponse(content=_to_response(notification).model_dump(mode="json"))
