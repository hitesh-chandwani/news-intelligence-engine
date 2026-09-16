"""Notification preferences API + page (issue #36; `design.md` §12).

Single-watch MVP: every endpoint here always operates on the Silver watch
(`nie.seed.run.SILVER_WATCH_SLUG`) -- there is no multi-watch selection
anywhere in this module, same precedent `nie.web.routers.watch` (#34) and
`nie.web.routers.context` (#35) set.

`NotificationPreference.watch_id` is the row's primary key itself (no
synthetic `id` column, `src/nie/models.py`), so `_get_preference_or_404`
looks the row up by `watch_id` directly. `seed()` always creates this row
(#14's `_seed_notification_preference`), so a missing row here means the
DB was migrated but never seeded -- same "never a 500" precedent
`_get_silver_watch` set in #34.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import Category, NotificationPreference, Watch
from nie.schemas import PreferenceResponse, PreferenceUpdate
from nie.seed.run import SILVER_WATCH_SLUG
from nie.web.deps import get_session

router = APIRouter()

# Same `Annotated` form `watch.py`/`context.py` use (FastAPI's recommended
# modern style) -- avoids ruff's B008 (function-call-in-default-argument)
# false positive on `Depends`.
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def _get_silver_watch(session: AsyncSession) -> Watch:
    """Fetch the Silver `Watch` row, or FastAPI's default `404` (a JSON
    `{"detail": ...}` body) when the DB has been migrated but never
    seeded -- identical to `watch.py`/`context.py`'s `_get_silver_watch`.
    """
    result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
    watch = result.scalar_one_or_none()
    if watch is None:
        raise HTTPException(status_code=404, detail=f"Watch {SILVER_WATCH_SLUG!r} not found")
    return watch


async def _get_preference_or_404(session: AsyncSession, watch: Watch) -> NotificationPreference:
    """Fetch `watch`'s `notification_preference` row, or `404` with a JSON
    `{"detail": ...}` body if it's somehow missing. `seed()` always creates
    this row, so in practice this only fires on a migrated-but-never-seeded
    DB -- same "never a 500" precedent `_get_silver_watch` set in #34.
    """
    result = await session.execute(
        select(NotificationPreference).where(NotificationPreference.watch_id == watch.id)
    )
    pref = result.scalar_one_or_none()
    if pref is None:
        raise HTTPException(
            status_code=404, detail=f"Notification preference for watch {watch.id} not found"
        )
    return pref


async def _all_categories(session: AsyncSession) -> list[Category]:
    """All 13 seeded `category` rows, ordered by `name`, for the
    multi-select's options. Queried directly from the `category` table --
    no `GET /categories` endpoint exists or is planned (`design.md` §12's
    table has none).
    """
    result = await session.execute(select(Category).order_by(Category.name))
    return list(result.scalars().all())


def _to_response(pref: NotificationPreference) -> PreferenceResponse:
    return PreferenceResponse(
        watch_id=pref.watch_id,
        min_importance=pref.min_importance,
        categories=list(pref.categories),
        channels=list(pref.channels),
    )


async def _respond(
    request: Request,
    session: AsyncSession,
    pref_response: PreferenceResponse,
) -> Response:
    """Negotiation shared by `GET`/`PATCH`: `HX-Request: true` -> `200
    text/html`, the re-rendered `partials/preferences_form.html` fragment
    (two-way, same `HX-Request` convention `watch.py`'s `_respond` and
    `context.py`'s `_respond_write` established); absent -> JSON
    (`PreferenceResponse`). `GET /preferences` layers a third branch (the
    full page) on top of this in `get_preferences` itself, mirroring
    `context.py`'s three-way `get_context`.
    """
    categories = await _all_categories(session)
    if request.headers.get("HX-Request"):
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "partials/preferences_form.html",
            {"preference": pref_response, "categories": categories},
        )
    return JSONResponse(content=pref_response.model_dump(mode="json"))


@router.get("/preferences")
async def get_preferences(request: Request, session: SessionDep) -> Response:
    """Three-way negotiation, same pattern `context.py`'s `get_context`
    established in #35 (one more axis than `watch.py`'s two-way, since
    there's no separate `/preferences/status`-style path to split page vs.
    API onto):

    - `HX-Request: true` -> `200 text/html`, the `partials/
      preferences_form.html` fragment
    - no `HX-Request` and `Accept` starts with `text/html` (a plain
      browser navigation) -> `200 text/html`, the full page
      (`preferences.html`)
    - anything else (including `httpx`'s default `*/*`) -> `200
      application/json`, `PreferenceResponse`
    """
    watch = await _get_silver_watch(session)
    pref = await _get_preference_or_404(session, watch)
    pref_response = _to_response(pref)

    if request.headers.get("HX-Request"):
        return await _respond(request, session, pref_response)

    accept = request.headers.get("accept", "")
    if accept.startswith("text/html"):
        categories = await _all_categories(session)
        templates: Jinja2Templates = request.app.state.templates
        return templates.TemplateResponse(
            request,
            "preferences.html",
            {"preference": pref_response, "categories": categories},
        )

    return JSONResponse(content=pref_response.model_dump(mode="json"))


@router.patch("/preferences")
async def update_preferences(
    request: Request, payload: PreferenceUpdate, session: SessionDep
) -> Response:
    """Update whichever of `min_importance`/`categories`/`channels` are set
    on `payload`, leaving the rest unchanged, and return the updated row
    in the same shape as `GET`. Two-way negotiation via `_respond`, same
    pattern `watch.py`'s `POST /watch/enable`/`disable` use.
    """
    watch = await _get_silver_watch(session)
    pref = await _get_preference_or_404(session, watch)

    if payload.min_importance is not None:
        pref.min_importance = payload.min_importance
    if payload.categories is not None:
        pref.categories = payload.categories
    if payload.channels is not None:
        pref.channels = payload.channels
    await session.commit()
    # `commit()` expires all attributes by default (`expire_on_commit=True`);
    # `_to_response` needs the refreshed values, same reasoning
    # `context.py`'s `create_context_item`/`update_context_item` use.
    await session.refresh(pref)

    return await _respond(request, session, _to_response(pref))
