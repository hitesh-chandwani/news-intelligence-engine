"""Context editor API + page (issue #35; `design.md` §12).

Single-watch MVP: every endpoint here always operates on the Silver watch
(`nie.seed.run.SILVER_WATCH_SLUG`) -- there is no multi-watch selection
anywhere in this module, same precedent `nie.web.routers.watch` set in
#34.

`ContextItem.kind` is either `"system"` (seeded by #14 from
`silver_context.md`, read-only through this API) or `"user"` (created,
edited, and deleted here). `ContextItemCreate` has no `kind` field, so
`kind="user"` is hardcoded server-side in `create_context_item` -- a
client can never request `kind="system"` through this API, and
`PATCH`/`DELETE` explicitly reject writes against a `kind="system"` row
with `403` (not `404` -- the row exists and was just visible via
`GET /context`; `403` says "read-only", `404` would falsely say "gone").
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import ContextItem, Watch
from nie.schemas import ContextItemCreate, ContextItemResponse, ContextItemUpdate
from nie.seed.run import SILVER_WATCH_SLUG
from nie.web.deps import get_session

router = APIRouter()

# Same `Annotated` form `watch.py` uses (FastAPI's recommended modern
# style) -- avoids ruff's B008 (function-call-in-default-argument) false
# positive on `Depends`.
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def _get_silver_watch(session: AsyncSession) -> Watch:
    """Fetch the Silver `Watch` row, or FastAPI's default `404` (a JSON
    `{"detail": ...}` body) when the DB has been migrated but never
    seeded -- identical to `watch.py`'s `_get_silver_watch`.
    """
    result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
    watch = result.scalar_one_or_none()
    if watch is None:
        raise HTTPException(status_code=404, detail=f"Watch {SILVER_WATCH_SLUG!r} not found")
    return watch


async def _get_context_item_or_404(session: AsyncSession, item_id: uuid.UUID) -> ContextItem:
    """Fetch a `ContextItem` row by id (either `kind`), or `404` with a
    JSON `{"detail": ...}` body when no row with that id exists at all.
    Callers that need to reject writes against `kind="system"` rows do
    that check themselves, after this lookup -- see the module docstring
    for why that's a separate `403`, not folded into this `404`.
    """
    result = await session.execute(select(ContextItem).where(ContextItem.id == item_id))
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail=f"Context item {item_id} not found")
    return item


def _require_user_item(item: ContextItem) -> None:
    """Raise `403` if `item` is a `kind="system"` row. Called by
    `PATCH`/`DELETE` after `_get_context_item_or_404` has already
    confirmed the row exists -- makes no change to `item`.
    """
    if item.kind == "system":
        raise HTTPException(status_code=403, detail="System context items are read-only")


def _to_response(item: ContextItem) -> ContextItemResponse:
    return ContextItemResponse(
        id=item.id,
        kind=item.kind,
        label=item.label,
        body=item.body,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


async def _user_items(session: AsyncSession, watch_id: uuid.UUID) -> list[ContextItem]:
    result = await session.execute(
        select(ContextItem)
        .where(ContextItem.watch_id == watch_id, ContextItem.kind == "user")
        .order_by(ContextItem.created_at)
    )
    return list(result.scalars().all())


async def _all_items(session: AsyncSession, watch_id: uuid.UUID) -> list[ContextItem]:
    result = await session.execute(
        select(ContextItem).where(ContextItem.watch_id == watch_id).order_by(ContextItem.created_at)
    )
    return list(result.scalars().all())


async def _respond_write(
    request: Request,
    session: AsyncSession,
    watch_id: uuid.UUID,
    json_status_code: int,
    body: ContextItemResponse | None,
) -> Response:
    """Two-way negotiation for `POST`/`PATCH`/`DELETE`, same `HX-Request`
    convention `watch.py`'s `_respond` established: present -> `200
    text/html`, the re-rendered `partials/context_user_items.html`
    fragment (only the user list changes via this API); absent -> JSON
    (`body is None` means `DELETE`'s `204` no-body response).
    """
    if request.headers.get("HX-Request"):
        user_items = await _user_items(session, watch_id)
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "partials/context_user_items.html",
            {"user_items": [_to_response(item) for item in user_items]},
        )
    if body is None:
        return Response(status_code=json_status_code)
    return JSONResponse(status_code=json_status_code, content=body.model_dump(mode="json"))


@router.get("/context")
async def get_context(request: Request, session: SessionDep) -> Response:
    """Three-way negotiation (issue #35 -- one more axis than `watch.py`'s
    two-way, since there's no separate `/context/status`-style path to
    split page vs. API onto):

    - `HX-Request: true` -> `200 text/html`, the `partials/
      context_lists.html` fragment (system + user blocks, no base layout)
    - no `HX-Request` and `Accept` starts with `text/html` (a plain
      browser navigation) -> `200 text/html`, the full page
      (`context.html`)
    - anything else (including `httpx`'s default `*/*`) -> `200
      application/json`, a JSON array of `ContextItemResponse`
    """
    watch = await _get_silver_watch(session)
    items = await _all_items(session, watch.id)
    system_items = [_to_response(item) for item in items if item.kind == "system"]
    user_items = [_to_response(item) for item in items if item.kind == "user"]

    templates: Jinja2Templates = request.app.state.templates
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request,
            "partials/context_lists.html",
            {"system_items": system_items, "user_items": user_items},
        )

    accept = request.headers.get("accept", "")
    if accept.startswith("text/html"):
        return templates.TemplateResponse(
            request,
            "context.html",
            {"system_items": system_items, "user_items": user_items},
        )

    all_items = [_to_response(item) for item in items]
    return JSONResponse(content=[item.model_dump(mode="json") for item in all_items])


@router.post("/context")
async def create_context_item(
    request: Request, payload: ContextItemCreate, session: SessionDep
) -> Response:
    """Create a `kind="user"` row. `kind` is hardcoded here, never taken
    from `payload` -- `ContextItemCreate` has no `kind` field at all, so
    there's nothing on the request body to accidentally trust.
    """
    watch = await _get_silver_watch(session)
    item = ContextItem(watch_id=watch.id, kind="user", label=payload.label, body=payload.body)
    session.add(item)
    await session.commit()
    # `commit()` expires all attributes by default (`expire_on_commit=True`);
    # `_to_response` needs the server-assigned `created_at`/`updated_at`, so
    # refresh before reading them back.
    await session.refresh(item)
    return await _respond_write(request, session, watch.id, 201, _to_response(item))


@router.patch("/context/{item_id}")
async def update_context_item(
    request: Request, item_id: uuid.UUID, payload: ContextItemUpdate, session: SessionDep
) -> Response:
    """Update `label`/`body` on an existing `kind="user"` row from
    whichever fields `payload` sets. `404` if no row with `item_id`
    exists at all; `403` (no change made) if it exists but is
    `kind="system"`.
    """
    item = await _get_context_item_or_404(session, item_id)
    _require_user_item(item)

    if payload.label is not None:
        item.label = payload.label
    if payload.body is not None:
        item.body = payload.body
    watch_id = item.watch_id
    await session.commit()
    # Same reasoning as `create_context_item`: `commit()` expires `item`'s
    # attributes, and `_to_response` needs the refreshed `updated_at`.
    await session.refresh(item)
    return await _respond_write(request, session, watch_id, 200, _to_response(item))


@router.delete("/context/{item_id}")
async def delete_context_item(
    request: Request, item_id: uuid.UUID, session: SessionDep
) -> Response:
    """Delete an existing `kind="user"` row. `404` if no row with
    `item_id` exists at all; `403` (no change made) if it exists but is
    `kind="system"`.
    """
    item = await _get_context_item_or_404(session, item_id)
    _require_user_item(item)

    watch_id = item.watch_id
    await session.delete(item)
    await session.commit()
    return await _respond_write(request, session, watch_id, 204, None)
