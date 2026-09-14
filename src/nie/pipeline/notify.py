"""Notification gate + payload builder (#30, `design.md` §5 stage 10,
§10, §8 FR-021, FR-022).

Neither `notify_gate` nor `build_notification_payload` is wired into
`STAGE_REGISTRY` or called from anywhere yet -- both are pure/DB-read-only
helper functions shipped ahead of, and independent from, their eventual
call site (#48), same "helper first, wiring later" precedent #22/#27 set.

`notify_gate` is synchronous with no DB access: it takes `reason` as an
explicit caller-supplied parameter rather than inferring "new-or-material"
itself -- see the issue's "Resolving 'how does the gate know
new-or-material'" section for why that inference deliberately does not
live here (filed as #48).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import (
    Category,
    Event,
    EventCategory,
    EventRelation,
    EventSource,
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
            relation=relation.relation,  # type: ignore[arg-type]
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
