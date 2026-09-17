"""Notification gate + payload builder + stage wiring (#30/#48,
`design.md` §5 stage 10, §10, §8 FR-019-FR-024).

`notify_gate`/`build_notification_payload` are pure/DB-read-only helper
functions shipped by #30 ahead of, and independent from, their eventual
call site -- `notify_stage`, added by #48, is that call site, wiring both
into `STAGE_REGISTRY` (`nie.pipeline.runner`) alongside the email (#31)/
Telegram (#32) channel senders.

`notify_gate` is synchronous with no DB access: it takes `reason` as an
explicit caller-supplied parameter rather than inferring "new-or-material"
itself -- `notify_stage`'s own selection query is what derives `reason`
(see its docstring).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.config import Settings
from nie.models import (
    Category,
    Event,
    EventCategory,
    EventRelation,
    EventSource,
    Notification,
    NotificationPreference,
    Source,
)

_IMPORTANCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def notify_gate(
    event: Event,
    reason: Literal["new-event", "material-update"] | None,
    category_slugs: Sequence[str],
    preference: NotificationPreference,
) -> bool:
    """Decide whether `event` should generate a notification for the watch
    `preference` belongs to, per `design.md` §5 stage 10. Pure, synchronous,
    no `session` parameter -- operates only on already-loaded values.

    All four conditions must hold:

    1. `reason is not None` -- the caller determined this run either
       created `event` (`"new-event"`) or materially updated it
       (`"material-update"`); `None` means neither happened this run, so
       there is nothing to notify about.
    2. `event.relevance not in (None, "irrelevant")` -- `None` means not
       yet scored (fail closed, don't notify, don't raise); `"irrelevant"`
       is an explicit scored verdict to suppress.
    3. `event.importance is not None` and ranks `>= preference.min_importance`
       on `_IMPORTANCE_ORDER` -- `None` means not yet scored (fail closed).
    4. `preference.categories` is empty (the documented "empty = all
       categories" convention), or shares at least one slug with
       `category_slugs`.

    `category_slugs` is supplied by the caller (loaded the same way
    `build_notification_payload` loads `categories`) rather than queried
    here, keeping this function DB-free and trivially unit-testable.
    """
    if reason is None:
        return False

    if event.relevance is None or event.relevance == "irrelevant":
        return False

    if event.importance is None:
        return False
    if _IMPORTANCE_ORDER[event.importance] < _IMPORTANCE_ORDER[preference.min_importance]:
        return False

    if preference.categories and not (set(preference.categories) & set(category_slugs)):
        return False

    return True


# ---------------------------------------------------------------------------
# build_notification_payload
# ---------------------------------------------------------------------------


class PayloadSource(BaseModel):
    title: str
    url: str
    source_name: str
    published_at: datetime | None


class PayloadRelatedEvent(BaseModel):
    event_id: uuid.UUID
    title: str
    event_date: datetime | None
    relation: Literal["precedes", "similar", "escalation-of", "context-for"]
    rationale: str


class NotificationPayload(BaseModel):
    event_id: uuid.UUID
    title: str
    fact_summary: str  # "what happened"
    interpretation: str  # "why it matters", part 1
    importance_rationale: str  # = event.impact_reason -- "why it matters", part 2;
    # there is no separate importance-rationale column, impact_reason is the
    # only free-text reasoning Event stores, so it fills both roles
    categories: list[str]  # category slugs, sorted
    importance: Literal["low", "medium", "high", "critical"]
    impact_direction: Literal["bullish", "bearish", "neutral", "unclear"]
    impact_confidence: Literal["low", "medium", "high"]
    related_events: list[PayloadRelatedEvent]
    sources: list[PayloadSource]


async def build_notification_payload(session: AsyncSession, event: Event) -> NotificationPayload:
    """Assemble the full `NotificationPayload` for `event`: its own columns
    plus three queries against tables it doesn't carry inline, per
    `design.md` §10's Content section (FR-021, FR-022).

    `categories`: every `category.slug` linked via `event_category` for
    `event.id`, sorted alphabetically -- same "sorted, deterministic"
    precedent #29 used for `shared_entities`.

    `related_events`: every `event_relation` row with `from_event_id ==
    event.id` -- the directionality #29 established ("this event ->
    historical candidate") -- joined to the candidate `Event` for
    `title`/`event_date`. Ordered by `event_date` descending (`NULLS
    LAST`), tie-broken by `to_event_id` ascending, for deterministic
    output/tests. `[]`, not an omitted key, when `event` has no outbound
    relations yet.

    `sources`: every `source` row joined via `event_source` for `event.id`
    (FR-022 -- "full source list", not just the triggering one). Ordered
    by `discovered_at` ascending, tie-broken by `id`, for deterministic
    output/tests.

    `fact_summary`/`interpretation`/`importance_rationale`
    (`= event.impact_reason`)/`importance`/`impact_direction`/
    `impact_confidence` are copied from `event` unchanged.
    """
    categories_result = await session.execute(
        select(Category.slug)
        .join(EventCategory, EventCategory.category_id == Category.id)
        .where(EventCategory.event_id == event.id)
        .order_by(Category.slug)
    )
    categories = list(categories_result.scalars())

    related_events_result = await session.execute(
        select(EventRelation, Event)
        .join(Event, Event.id == EventRelation.to_event_id)
        .where(EventRelation.from_event_id == event.id)
        .order_by(Event.event_date.desc().nulls_last(), EventRelation.to_event_id)
    )
    related_events = [
        PayloadRelatedEvent(
            event_id=candidate_event.id,
            title=candidate_event.title,
            event_date=candidate_event.event_date,
            relation=relation.relation,
            rationale=relation.rationale,
        )
        for relation, candidate_event in related_events_result.all()
    ]

    sources_result = await session.execute(
        select(Source)
        .join(EventSource, EventSource.source_id == Source.id)
        .where(EventSource.event_id == event.id)
        .order_by(Source.discovered_at, Source.id)
    )
    sources = [
        PayloadSource(
            title=source.title,
            url=source.url,
            source_name=source.source_name,
            published_at=source.published_at,
        )
        for source in sources_result.scalars()
    ]

    assert event.importance is not None
    assert event.impact_direction is not None
    assert event.impact_confidence is not None
    assert event.impact_reason is not None

    return NotificationPayload(
        event_id=event.id,
        title=event.title,
        fact_summary=event.fact_summary,
        interpretation=event.interpretation,
        importance_rationale=event.impact_reason,
        categories=categories,
        importance=event.importance,  # type: ignore[arg-type]
        impact_direction=event.impact_direction,  # type: ignore[arg-type]
        impact_confidence=event.impact_confidence,  # type: ignore[arg-type]
        related_events=related_events,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# notify_stage
# ---------------------------------------------------------------------------


async def notify_stage(session: AsyncSession) -> dict[str, int]:
    """Notify stage (#48, `design.md` §5 stage 10). A `StageFn` (per #18)
    with the plain `StageFn` signature every other real stage but
    `score_stage`/`relate_stage` uses -- no `client` parameter, since
    `design.md` §5 stage 10's Model column is `--` (no LLM call here).

    **Candidate selection / `reason`.** A candidate is any `Event` row
    where either (a) no `Notification` row exists yet for `event.id` --
    `reason = "new-event"` -- or (b) `Event.last_material_update_at` is
    later than the `created_at` of that event's most recent `Notification`
    row -- `reason = "material-update"`. Implemented as one query: a
    `GROUP BY Notification.event_id` subquery of each event's latest
    `Notification.created_at` (`last_notified_at`), left-outer-joined onto
    `Event`, filtered to `last_notified_at IS NULL OR
    Event.last_material_update_at > last_notified_at`. An event already
    notified for its current `last_material_update_at` is excluded by
    this WHERE clause itself -- `notify_gate` is never even called for
    it, which is what makes a second call idempotent (#48's "Idempotency"
    criterion): with no `last_material_update_at` change between calls,
    the second call's query selects nothing for that event. Ordered by
    `Event.id` for deterministic iteration/test output.

    **Preference lookup.** For each candidate, `preference =
    await session.get(NotificationPreference, event.watch_id)` --
    `watch_id` is that table's primary key (`models.py`). `None` (the
    watch has no preference row) skips the event silently: no gate call,
    no error, same "seed() always creates this row; a missing one is
    never a crash" precedent `_get_preference_or_404`
    (`src/nie/web/routers/preferences.py`) sets, applied here as "skip"
    since this is a background stage, not an HTTP handler.

    **Gate.** `category_slugs` is loaded the same way
    `build_notification_payload` loads `categories` (`Category.slug`
    joined via `event_category`), then `notify_gate(event, reason,
    category_slugs, preference)` decides. `False` skips the event
    entirely -- no payload build, no channel calls, no `Notification` row.

    **On a pass:** `build_notification_payload(session, event)` builds the
    payload, then every channel in `preference.channels` is attempted
    independently -- `"email"` via `send_email(payload, settings)` (caught:
    `EmailSendError`), `"telegram"` via `send_telegram_message(payload,
    settings)` (caught: `TelegramSendError`), `"inapp"` with no send call
    at all (the `Notification` row's own existence *is* the inapp
    delivery, per `design.md` §10). `send_email`/`send_telegram_message`
    are imported inside this function rather than at module level --
    `nie.notifications.email`/`nie.notifications.telegram` both import
    `NotificationPayload` from this module, so a top-level import here
    would be a circular import; a local import, executed only once this
    module has finished defining everything, breaks the cycle regardless
    of which of the two modules happens to be imported first.

    **Failure isolation.** An `EmailSendError`/`TelegramSendError` from
    one channel is caught right where that channel is attempted -- it
    only leaves that one channel out of `channels_sent`; the other
    channel is still attempted, the `Notification` row is still inserted,
    and the loop still continues to the next candidate event. No
    exception from one event's processing ever propagates out of
    `notify_stage`, same per-item isolation precedent
    `extract_stage`/`synthesize_stage`/`score_stage` set for their own
    exception types.

    A `Notification(watch_id=event.watch_id, event_id=event.id,
    reason=reason, payload=payload.model_dump(mode="json"),
    channels_sent=channels_sent)` row is `session.add()`-ed
    **unconditionally** once the gate passes -- regardless of whether any
    channel is configured or any send succeeds (today's seeded default is
    `channels=[]`, so a passing gate still adds a row with
    `channels_sent == []` -- `src/nie/seed/run.py`'s
    `_seed_notification_preference`). `channels_sent` contains
    `"email"`/`"telegram"` iff that channel was in `preference.channels`
    and its send did not raise, and `"inapp"` iff it was in
    `preference.channels`.

    Does not call `session.commit()` -- only `session.add()`s
    `Notification` rows, matching every other stage; the runner (#18)
    commits once after the stage returns.

    Returns `{"notified": N, "gated_out": G, "skipped_no_preference": S}`
    -- `notified` counts candidates for which a `Notification` row was
    added this call (regardless of `channels_sent` contents); `gated_out`
    counts candidates `notify_gate` returned `False` for; `S` counts
    candidates skipped for a missing `NotificationPreference` row.
    """
    from nie.notifications.email import EmailSendError, send_email
    from nie.notifications.telegram import TelegramSendError, send_telegram_message

    settings = Settings()

    last_notification = (
        select(
            Notification.event_id.label("event_id"),
            func.max(Notification.created_at).label("last_notified_at"),
        )
        .group_by(Notification.event_id)
        .subquery()
    )

    candidates_result = await session.execute(
        select(Event, last_notification.c.last_notified_at)
        .outerjoin(last_notification, last_notification.c.event_id == Event.id)
        .where(
            or_(
                last_notification.c.last_notified_at.is_(None),
                Event.last_material_update_at > last_notification.c.last_notified_at,
            )
        )
        .order_by(Event.id)
    )
    candidates = candidates_result.all()

    notified_count = 0
    gated_out_count = 0
    skipped_no_preference_count = 0

    for event, last_notified_at in candidates:
        reason: Literal["new-event", "material-update"] = (
            "new-event" if last_notified_at is None else "material-update"
        )

        preference = await session.get(NotificationPreference, event.watch_id)
        if preference is None:
            skipped_no_preference_count += 1
            continue

        category_slugs_result = await session.execute(
            select(Category.slug)
            .join(EventCategory, EventCategory.category_id == Category.id)
            .where(EventCategory.event_id == event.id)
        )
        category_slugs = list(category_slugs_result.scalars())

        if not notify_gate(event, reason, category_slugs, preference):
            gated_out_count += 1
            continue

        payload = await build_notification_payload(session, event)

        channels_sent: list[str] = []
        for channel in preference.channels:
            if channel == "email":
                try:
                    await send_email(payload, settings)
                except EmailSendError:
                    continue
                channels_sent.append("email")
            elif channel == "telegram":
                try:
                    await send_telegram_message(payload, settings)
                except TelegramSendError:
                    continue
                channels_sent.append("telegram")
            elif channel == "inapp":
                channels_sent.append("inapp")

        session.add(
            Notification(
                watch_id=event.watch_id,
                event_id=event.id,
                reason=reason,
                payload=payload.model_dump(mode="json"),
                channels_sent=channels_sent,
            )
        )
        notified_count += 1

    return {
        "notified": notified_count,
        "gated_out": gated_out_count,
        "skipped_no_preference": skipped_no_preference_count,
    }
