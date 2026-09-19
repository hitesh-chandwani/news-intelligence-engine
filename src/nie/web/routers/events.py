"""Event timeline + detail API + page (issue #37; `design.md` §12, §13).

Single-watch MVP: `GET /events` always scopes to the Silver watch
(`nie.seed.run.SILVER_WATCH_SLUG`) -- there is no multi-watch selection
anywhere in this module, same precedent `nie.web.routers.watch` (#34),
`context.py` (#35), and `preferences.py` (#36) set. `GET /events/{id}` is
not scoped to the Silver watch in its lookup (an id is already globally
unique), but MVP has exactly one watch so this is moot in practice.

Both `GET` endpoints are read-only with no JSON-body forms, so neither
the `hx-ext="json-enc"` pitfall nor the checkbox-group serialization
pitfall #35/#36 hit applies here: there is no such form, and the filter
form's category control is a native `<select multiple>` (serialized by
`hx-get` as a plain, always-present query string), not a checkbox group.

`POST /events/{id}/feedback` (#39) lives here rather than in its own
`nie.web.routers.feedback` module: unlike every other `POST` this app has
added so far, its `HX-Request` response has to re-render a *rich* fragment
(`partials/event_detail.html`, with sources/related events/categories
already loaded) when submitted from the event detail page, and this
module already owns `_get_event_or_404`/`_build_detail` -- the exact
helpers that requires. Duplicating that machinery in a new file just to
keep "one router per resource" would be pure repetition; extending this
module was the precedent the issue's Constraints left as an explicit
option, and this is the codebase's existing "don't import another
router's private helpers, but do avoid outright duplicating a big one"
tradeoff. It is also bodyless (`?verdict=<value>`, no JSON body), so
neither #35's `hx-ext="json-enc"` requirement nor #36's checkbox bug class
applies to it either.

`DELETE /events/{id}/feedback/{feedback_id}` (#51) lives here too, next
to `submit_event_feedback`, for the same "already owns `_get_event_or_404`/
`_build_detail`" reasoning: an "Undo" control shown alongside the
confirmation `submit_event_feedback` just rendered has to re-render that
same rich fragment (or `partials/notification_list.html`, via the same
`HX-Target` branching) once the just-created row is deleted.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.feedback import (
    EventNotFoundError,
    FeedbackNotFoundError,
    UnknownVerdictError,
    submit_feedback,
    withdraw_feedback,
)
from nie.models import (
    Category,
    Event,
    EventCategory,
    EventRelation,
    EventSource,
    Feedback,
    Notification,
    Source,
    Watch,
)
from nie.notifications.inbox import list_notifications
from nie.schemas import (
    EventDetail,
    EventSourceRef,
    EventSummary,
    FeedbackResponse,
    NotificationResponse,
    RelatedEventRef,
)
from nie.seed.run import SILVER_WATCH_SLUG
from nie.web.deps import get_session

router = APIRouter()

# Same `Annotated` form `watch.py`/`context.py`/`preferences.py` use
# (FastAPI's recommended modern style) -- avoids ruff's B008
# (function-call-in-default-argument) false positive on `Depends`.
SessionDep = Annotated[AsyncSession, Depends(get_session)]

# `GET /events` returns at most this many rows, most-recent-first -- no
# pagination controls in this task (see issue #37's Out of scope).
_MAX_EVENTS = 200

# The exact value sets `nie.models.Event`'s `CheckConstraint`s allow for
# `importance`/`relevance`.
_IMPORTANCE_VALUES = {"low", "medium", "high", "critical"}
_RELEVANCE_VALUES = {"irrelevant", "low", "medium", "high"}


def _normalize_enum_query(value: str | None, allowed: set[str], field_name: str) -> str | None:
    """Validate an `importance`/`relevance` query param against `allowed`,
    raising the same `422` FastAPI's own enum validation would give for a
    value outside that set.

    Blank (`""`) is treated the same as omitted (`None`) -- "no filter" --
    rather than an invalid value: `events.html`'s filter `<select>`s each
    carry a blank "Any" `<option>` (native `<select>`s, unlike checkboxes,
    have no way to omit a field entirely; leaving it unselected still
    submits `name=`), and this is the one normalization point that turns
    that into "not filtered on this field", same intent as `importance`/
    `relevance` simply being absent from the query string.
    """
    if value is None or value == "":
        return None
    if value not in allowed:
        raise HTTPException(
            status_code=422, detail=f"{field_name} must be one of {sorted(allowed)}"
        )
    return value


def _parse_optional_datetime(value: str | None, field_name: str) -> datetime | None:
    """Parse `date_from`/`date_to` from an ISO 8601 string, `422` on an
    unparseable non-blank value. Blank (`""`) is "no filter", same
    reasoning as `_normalize_enum_query` -- `events.html`'s `date`
    `<input>`s submit `name=` when left empty, same as the `<select>`s.
    """
    if value is None or value == "":
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"{field_name} must be an ISO 8601 datetime"
        ) from exc


async def _get_silver_watch(session: AsyncSession) -> Watch:
    """Fetch the Silver `Watch` row, or FastAPI's default `404` (a JSON
    `{"detail": ...}` body) when the DB has been migrated but never
    seeded -- identical to `watch.py`/`context.py`/`preferences.py`'s
    `_get_silver_watch`.
    """
    result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
    watch = result.scalar_one_or_none()
    if watch is None:
        raise HTTPException(status_code=404, detail=f"Watch {SILVER_WATCH_SLUG!r} not found")
    return watch


async def _all_categories(session: AsyncSession) -> list[Category]:
    """All seeded `category` rows, ordered by `name`, for the filter
    form's multi-select options -- same helper `preferences.py` defines
    for its own multi-select, duplicated here rather than imported across
    router modules (neither router imports the other's private helpers).
    """
    result = await session.execute(select(Category).order_by(Category.name))
    return list(result.scalars().all())


async def _categories_by_event_id(
    session: AsyncSession, event_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[str]]:
    """Bulk-fetch category slugs for every id in `event_ids` in one query,
    grouped by `event_id` -- avoids an N+1 query per timeline row.
    """
    if not event_ids:
        return {}
    result = await session.execute(
        select(EventCategory.event_id, Category.slug)
        .join(Category, Category.id == EventCategory.category_id)
        .where(EventCategory.event_id.in_(event_ids))
    )
    by_event: dict[uuid.UUID, list[str]] = {event_id: [] for event_id in event_ids}
    for event_id, slug in result.all():
        by_event[event_id].append(slug)
    return by_event


def _to_summary(event: Event, categories: list[str]) -> EventSummary:
    return EventSummary(
        id=event.id,
        title=event.title,
        event_date=event.event_date,
        discovered_at=event.discovered_at,
        relevance=event.relevance,
        importance=event.importance,
        impact_direction=event.impact_direction,
        impact_confidence=event.impact_confidence,
        categories=categories,
    )


async def _list_events(
    session: AsyncSession,
    watch: Watch,
    category: list[str] | None,
    importance: str | None,
    relevance: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
) -> list[EventSummary]:
    """Every `event` row for `watch`, filtered per the query params below
    (all AND'ed together), ordered by `event_date desc` (a `NULL`
    `event_date` sorts last), tie-broken by `discovered_at desc`, capped
    at the `_MAX_EVENTS` most recent matching rows.

    **Edge case (by design, not a bug):** stage 7 (synthesize) creates an
    `event` row before stage 8 (score) fills in `relevance`/`importance`/
    `impact_*`, and `event_date` itself can be `NULL` (`design.md` §4:
    "when it happened", not always knowable). Such rows are included in
    an *unfiltered* `GET /events`, but standard SQL `NULL` comparison
    semantics mean a `NULL` on a filtered column (`importance`,
    `relevance`, or `event_date` under `date_from`/`date_to`) never
    satisfies an `=`/`>=`/`<=` comparison, so those rows silently drop out
    of any filtered result. No special-case handling is added for this --
    it falls straight out of how SQL `NULL` already behaves.
    """
    stmt = select(Event).where(Event.watch_id == watch.id)

    if category:
        # An event matches if it has *any* of the given category slugs
        # (OR'ed together, consistent with an event having one or more
        # categories, FR-011). An `IN` subquery (rather than a join)
        # keeps this an OR without risking duplicate `Event` rows in the
        # outer result when an event matches more than one given slug.
        matching_event_ids = (
            select(EventCategory.event_id)
            .join(Category, Category.id == EventCategory.category_id)
            .where(Category.slug.in_(category))
        )
        stmt = stmt.where(Event.id.in_(matching_event_ids))
    if importance is not None:
        stmt = stmt.where(Event.importance == importance)
    if relevance is not None:
        stmt = stmt.where(Event.relevance == relevance)
    if date_from is not None:
        stmt = stmt.where(Event.event_date >= date_from)
    if date_to is not None:
        stmt = stmt.where(Event.event_date <= date_to)

    stmt = stmt.order_by(
        Event.event_date.desc().nulls_last(), Event.discovered_at.desc()
    ).limit(_MAX_EVENTS)

    result = await session.execute(stmt)
    events = list(result.scalars().all())
    categories_by_event = await _categories_by_event_id(session, [event.id for event in events])
    return [_to_summary(event, categories_by_event.get(event.id, [])) for event in events]


async def _get_event_or_404(session: AsyncSession, event_id: uuid.UUID) -> Event:
    """Fetch an `Event` row by id, or `404` with a JSON `{"detail": ...}`
    body when no row with that id exists -- same "no 500" precedent
    `_get_silver_watch` set in #34. Not scoped to the Silver watch: an id
    is already globally unique.
    """
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
    return event


async def _event_sources(session: AsyncSession, event_id: uuid.UUID) -> list[EventSourceRef]:
    """Every `Source` linked to `event_id` via `EventSource`, each with
    its `url` so the UI can link out to the original article (FR-022).
    """
    result = await session.execute(
        select(Source)
        .join(EventSource, EventSource.source_id == Source.id)
        .where(EventSource.event_id == event_id)
        .order_by(EventSource.linked_at)
    )
    sources = result.scalars().all()
    return [
        EventSourceRef(
            id=source.id,
            url=source.url,
            title=source.title,
            source_name=source.source_name,
            published_at=source.published_at,
        )
        for source in sources
    ]


async def _related_events(session: AsyncSession, event_id: uuid.UUID) -> list[RelatedEventRef]:
    """`EventRelation` rows in **both** directions for `event_id`: rows
    where it's `from_event_id` (`direction="outgoing"`, the *other*
    event's title is the `to_event_id` row's title) and rows where it's
    `to_event_id` (`direction="incoming"`, the *other* event's title is
    the `from_event_id` row's title). An event related to two others (one
    each way) returns both, not just one.
    """
    outgoing_result = await session.execute(
        select(EventRelation.to_event_id, EventRelation.relation, Event.title)
        .join(Event, Event.id == EventRelation.to_event_id)
        .where(EventRelation.from_event_id == event_id)
    )
    outgoing = [
        RelatedEventRef(event_id=other_id, title=title, relation=relation, direction="outgoing")
        for other_id, relation, title in outgoing_result.all()
    ]

    incoming_result = await session.execute(
        select(EventRelation.from_event_id, EventRelation.relation, Event.title)
        .join(Event, Event.id == EventRelation.from_event_id)
        .where(EventRelation.to_event_id == event_id)
    )
    incoming = [
        RelatedEventRef(event_id=other_id, title=title, relation=relation, direction="incoming")
        for other_id, relation, title in incoming_result.all()
    ]

    return outgoing + incoming


async def _build_detail(session: AsyncSession, event: Event) -> EventDetail:
    categories_by_event = await _categories_by_event_id(session, [event.id])
    sources = await _event_sources(session, event.id)
    related_events = await _related_events(session, event.id)
    return EventDetail(
        id=event.id,
        title=event.title,
        fact_summary=event.fact_summary,
        interpretation=event.interpretation,
        event_date=event.event_date,
        discovered_at=event.discovered_at,
        relevance=event.relevance,
        importance=event.importance,
        impact_direction=event.impact_direction,
        impact_reason=event.impact_reason,
        impact_confidence=event.impact_confidence,
        entities=list(event.entities),
        categories=categories_by_event.get(event.id, []),
        sources=sources,
        related_events=related_events,
    )


@router.get("/events")
async def list_events(
    request: Request,
    session: SessionDep,
    category: Annotated[list[str] | None, Query()] = None,
    importance: Annotated[str | None, Query()] = None,
    relevance: Annotated[str | None, Query()] = None,
    date_from: Annotated[str | None, Query()] = None,
    date_to: Annotated[str | None, Query()] = None,
) -> Response:
    """Three-way negotiation, same pattern `context.py`'s `get_context`
    (#35) established:

    - `HX-Request: true` -> `200 text/html`, the `partials/
      event_list.html` fragment (list only)
    - no `HX-Request` and `Accept` starts with `text/html` (a plain
      browser navigation) -> `200 text/html`, the full page
      (`events.html`, includes the filter form + the same list fragment)
    - anything else (including `httpx`'s default `*/*`) -> `200
      application/json`, a JSON array of `EventSummary`

    `importance`/`relevance`/`date_from`/`date_to` are taken as plain
    strings (rather than typed directly as e.g. `Literal[...] | None` or
    `datetime | None`) so `_normalize_enum_query`/`_parse_optional_datetime`
    can treat a blank value the same as an omitted one -- see their
    docstrings for why a native HTML `<select>`/`<input type="date">`
    needs that (unlike a checkbox, neither can be omitted from a
    submission by being left at its default).

    The `HX-Request` branch passes `filters` into `partials/
    event_list.html`, same as the full-page branch, so its `#event-list`
    polling loop (`hx-trigger="every 15s [!document.hidden]"`, #66,
    the #63/#64/#65 pattern) can build its own re-rendered `hx-get` with
    the currently applied filters baked into the query string -- without
    this, the filters would only survive the first poll tick after a
    filtered page load/submission, not every tick after that.
    """
    watch = await _get_silver_watch(session)
    importance_value = _normalize_enum_query(importance, _IMPORTANCE_VALUES, "importance")
    relevance_value = _normalize_enum_query(relevance, _RELEVANCE_VALUES, "relevance")
    date_from_value = _parse_optional_datetime(date_from, "date_from")
    date_to_value = _parse_optional_datetime(date_to, "date_to")
    events = await _list_events(
        session, watch, category, importance_value, relevance_value, date_from_value, date_to_value
    )

    templates: Jinja2Templates = request.app.state.templates
    filters = {
        "category": category or [],
        "importance": importance_value,
        "relevance": relevance_value,
        "date_from": date_from_value,
        "date_to": date_to_value,
    }

    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request, "partials/event_list.html", {"events": events, "filters": filters}
        )

    accept = request.headers.get("accept", "")
    if accept.startswith("text/html"):
        categories = await _all_categories(session)
        return templates.TemplateResponse(
            request,
            "events.html",
            {"events": events, "categories": categories, "filters": filters},
        )

    return JSONResponse(content=[event.model_dump(mode="json") for event in events])


@router.get("/events/{event_id}")
async def get_event(request: Request, event_id: uuid.UUID, session: SessionDep) -> Response:
    """Three-way negotiation, same pattern as `list_events` above / and
    `context.py`'s `get_context` (#35):

    - `HX-Request: true` -> `200 text/html`, the `partials/
      event_detail.html` fragment
    - no `HX-Request` and `Accept` starts with `text/html` (a plain
      browser navigation) -> `200 text/html`, the full page
      (`event_detail.html`)
    - anything else (including `httpx`'s default `*/*`) -> `200
      application/json`, `EventDetail`

    `404` with a JSON `{"detail": ...}` body when no `event` row with
    `event_id` exists.
    """
    event = await _get_event_or_404(session, event_id)
    detail = await _build_detail(session, event)

    templates: Jinja2Templates = request.app.state.templates
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse(
            request, "partials/event_detail.html", {"event": detail}
        )

    accept = request.headers.get("accept", "")
    if accept.startswith("text/html"):
        return templates.TemplateResponse(request, "event_detail.html", {"event": detail})

    return JSONResponse(content=detail.model_dump(mode="json"))


def _to_feedback_response(feedback: Feedback) -> FeedbackResponse:
    return FeedbackResponse(
        id=feedback.id,
        watch_id=feedback.watch_id,
        event_id=feedback.event_id,
        verdict=feedback.verdict,  # type: ignore[arg-type]
        note=feedback.note,
        created_at=feedback.created_at,
    )


def _notification_to_response(notification: Notification) -> NotificationResponse:
    """Same shape as `nie.web.routers.notifications._to_response` --
    duplicated here rather than imported, same "don't import another
    router's private helpers" convention `_all_categories` already
    established between this module and `preferences.py`.
    """
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


@router.post("/events/{event_id}/feedback")
async def submit_event_feedback(
    request: Request,
    event_id: uuid.UUID,
    verdict: Annotated[str, Query()],
    session: SessionDep,
    note: Annotated[str | None, Form()] = None,
) -> Response:
    """Persist one feedback verdict (and optional note, #50) against
    `event_id` (FR-025), via `nie.feedback.submit_feedback`. `verdict` is a
    required query param (e.g. `?verdict=useful`), modeled on `POST
    /notifications/{id}/read`'s bodyless `<button hx-post="...">` pattern
    (see the module docstring for why #35/#36's bug classes don't apply);
    `note` (#50) is an optional form field read from the request body
    (`application/x-www-form-urlencoded`, htmx's default encoding, via
    each button's `hx-include` pointing at a shared `<textarea>`) -- a
    route can mix a `Query()` param and a `Form()` param, so `verdict`
    stays exactly as #39 shipped it while `note` moves through the body.
    `submit_feedback` owns all `note` normalization (blank/whitespace-only
    -> `None`, trim otherwise); this route passes it through unchanged.

    `submit_feedback` is HTTP-agnostic and raises a `ValueError` subclass
    per failure mode; that translation happens here, same pattern
    `mark_notification_read` (#38) uses for `mark_read`'s `ValueError`:

    - `UnknownVerdictError` (verdict not one of the 5
      `Feedback.verdict`-allowed values) -> `422`
    - `EventNotFoundError` (no `event` row with `event_id`) -> `404`

    (Unaffected by whether `note` was supplied -- a bad `verdict` or
    missing `event_id` behaves identically either way.)

    On success, response negotiation follows the `HX-Request` convention
    every other write route in this app uses, but the feedback buttons
    render on two different pages with two different `hx-target`s
    (`#event-detail` on the event detail page, `#notification-list` in the
    inbox) -- htmx automatically sends the target element's id as the
    `HX-Target` request header on every request it makes, so that header
    (not an extra query param the issue's Constraints don't call for)
    decides which fragment to re-render:

    - `HX-Target: notification-list` -> re-renders `partials/
      notification_list.html` for the feedback's watch (same fragment
      `mark_notification_read` re-renders), with `feedback_submitted_event_id`/
      `feedback_submitted_verdict`/`feedback_submitted_note`/
      `feedback_submitted_id` in the template context so the submitting
      notification's row shows a confirmation (with the note, when
      non-blank) and an "Undo" control (#51) targeting the just-created
      row's id.
    - anything else while `HX-Request: true` (including `HX-Target:
      event-detail`, or no `HX-Target` at all) -> re-renders `partials/
      event_detail.html` for `event_id`, with `feedback_submitted`/
      `feedback_submitted_note`/`feedback_submitted_id` in the template
      context so the confirmation and its "Undo" control (#51) render
      there instead.
    - no `HX-Request` header -> `200 application/json`, `FeedbackResponse`.
    """
    try:
        feedback = await submit_feedback(session, event_id, verdict, note)
    except UnknownVerdictError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EventNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    templates: Jinja2Templates = request.app.state.templates
    if request.headers.get("HX-Request"):
        if request.headers.get("HX-Target") == "notification-list":
            notifications = await list_notifications(session, feedback.watch_id)
            responses = [_notification_to_response(item) for item in notifications]
            return templates.TemplateResponse(
                request,
                "partials/notification_list.html",
                {
                    "notifications": responses,
                    "feedback_submitted_event_id": feedback.event_id,
                    "feedback_submitted_verdict": feedback.verdict,
                    "feedback_submitted_note": feedback.note,
                    "feedback_submitted_id": feedback.id,
                },
            )

        event = await _get_event_or_404(session, feedback.event_id)
        detail = await _build_detail(session, event)
        return templates.TemplateResponse(
            request,
            "partials/event_detail.html",
            {
                "event": detail,
                "feedback_submitted": feedback.verdict,
                "feedback_submitted_note": feedback.note,
                "feedback_submitted_id": feedback.id,
            },
        )

    return JSONResponse(content=_to_feedback_response(feedback).model_dump(mode="json"))


@router.delete("/events/{event_id}/feedback/{feedback_id}")
async def withdraw_event_feedback(
    request: Request,
    event_id: uuid.UUID,
    feedback_id: uuid.UUID,
    session: SessionDep,
) -> Response:
    """Withdraw ("Undo", #51) a just-submitted `Feedback` row via
    `nie.feedback.withdraw_feedback`, hard-deleting it (`design.md` §4's
    `feedback` schema is unchanged -- no soft delete).

    `withdraw_feedback` is HTTP-agnostic and raises `FeedbackNotFoundError`
    (a `ValueError` subclass) when no `feedback` row with `feedback_id`
    exists, or when it exists but its `event_id` doesn't match the path's
    `event_id`; that translates to `404` here, same pattern
    `submit_event_feedback` uses for `EventNotFoundError`/
    `UnknownVerdictError` and `_get_context_item_or_404`/
    `_get_event_or_404` use for their own not-found cases. A double-
    withdraw (already-deleted `feedback_id`, e.g. a double-click on Undo)
    finds nothing on the second call and gets the same `404`, not a
    silent no-op success.

    On success, response negotiation follows the same `HX-Request`/
    `HX-Target` convention `submit_event_feedback` documents (htmx sends
    the target element's id as the `HX-Target` header):

    - `HX-Target: notification-list` -> re-renders `partials/
      notification_list.html` for the Silver watch (single-watch MVP,
      same `_get_silver_watch` resolution every other router in this
      module uses), with no `feedback_submitted_event_id` in the
      template context -- the withdrawn row is gone, so that
      notification's confirmation clears.
    - anything else while `HX-Request: true` (including `HX-Target:
      event-detail`, or no `HX-Target` at all) -> re-renders `partials/
      event_detail.html` for `event_id`, with no `feedback_submitted` in
      the template context -- the fragment renders back to its plain
      5-verdict-button state.
    - no `HX-Request` header -> `204 No Content`, empty body, same
      convention `DELETE /context/{item_id}` (#35) already uses.
    """
    try:
        await withdraw_feedback(session, event_id, feedback_id)
    except FeedbackNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if request.headers.get("HX-Request"):
        templates: Jinja2Templates = request.app.state.templates
        if request.headers.get("HX-Target") == "notification-list":
            watch = await _get_silver_watch(session)
            notifications = await list_notifications(session, watch.id)
            responses = [_notification_to_response(item) for item in notifications]
            return templates.TemplateResponse(
                request,
                "partials/notification_list.html",
                {"notifications": responses},
            )

        event = await _get_event_or_404(session, event_id)
        detail = await _build_detail(session, event)
        return templates.TemplateResponse(
            request, "partials/event_detail.html", {"event": detail}
        )

    return Response(status_code=204)
