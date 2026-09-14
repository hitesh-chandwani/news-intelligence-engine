"""Context bundle builder (#27, `design.md` §6 "Context bundle (FR-013)").

`build_context_bundle` is a plain async function, **not** a
`STAGE_REGISTRY` `StageFn` -- pure retrieval, no LLM call. It assembles
everything stage 8 (score, #28) needs to build its prompt: the watch's
`context_item` rows, event-relation-linked and vector-nearest historical
events (with their shared entities), and a rolling feedback summary
bucketed by category.

`score.py` is the module `design.md` §15 already assigns to stage 8, so
`build_context_bundle` lives here ahead of #28's `score_stage` (plus
`score.md`), the same way `match.py` (#22) holds `find_candidate_events`
ahead of any stage using it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import (
    Category,
    ContextItem,
    Event,
    EventCategory,
    EventRelation,
    Feedback,
    Watch,
)
from nie.pipeline.match import CONTEXT_EVENT_LIMIT, find_nearest_events

# Not a `Settings`/`.env` field -- `design.md` §14's config table has no
# feedback-window entry (only `DEDUP_WINDOW_DAYS`, a different setting
# serving a different, per-run recall purpose). Same "not a config field"
# precedent as `MATCH_CANDIDATE_LIMIT`/`STUB_FIXTURES_DIR`. `feedback_window_days`
# stays a keyword argument (default this constant) so callers/tests can
# override it without touching the constant itself.
FEEDBACK_WINDOW_DAYS = 30


class RelatedEvent(NamedTuple):
    event: Event
    matched_via: frozenset[str]
    distance: float | None
    shared_entities: list[str]


class FeedbackBucket(NamedTuple):
    category_slug: str
    verdict: str
    # `count` shadows `tuple.count` (the built-in occurrence-counting
    # method) -- a known mypy limitation with `NamedTuple` field names
    # (python/mypy#1021), not a real type error: the field works
    # correctly at runtime (see `tests/test_pipeline_score.py`).
    count: int  # type: ignore[assignment]


@dataclass(frozen=True)
class ContextBundle:
    system_context: list[ContextItem]
    user_context: list[ContextItem]
    related_events: list[RelatedEvent]
    feedback_summary: list[FeedbackBucket]


async def build_context_bundle(
    session: AsyncSession,
    watch: Watch,
    event: Event,
    *,
    vector_limit: int = CONTEXT_EVENT_LIMIT,
    feedback_window_days: int = FEEDBACK_WINDOW_DAYS,
) -> ContextBundle:
    """Assemble the retrieval bundle stage 8 (#28) needs to build its
    score prompt.

    `system_context`/`user_context` are `watch`'s `context_item` rows
    (#6), split by `kind`.

    `related_events` unions two sources, deduping by event id:

    - Every `event_relation` row linking `event` in either direction
      (relations are directional; either side counts as "linked" for
      context purposes), tagged `f"relation:{row.relation}"` per linking
      row -- more than one row can link the same pair, and all of them
      contribute a tag.
    - `find_nearest_events` (`match.py`, #27), capped at `vector_limit`,
      tagged `"vector"`, contributing `distance`.

    An event reachable by both appears once, with `matched_via` the union
    of both tags and `distance` populated from the vector search. An
    event with zero `event_relation` rows (e.g. its first score pass,
    before relate/#29 has run) still returns whatever `find_nearest_events`
    finds, without error -- `related_events` may be `[]`.

    `shared_entities` per related event is the sorted intersection of
    `event.entities` and that related event's own `entities`.

    `feedback_summary` buckets `feedback` rows for `watch` by
    `(category.slug, verdict)` via the `event_category` join
    (`feedback.event_id -> event_category.event_id -> category.id`),
    counting only rows with `created_at >= now - feedback_window_days`.
    Because `event_category` is many-to-many, a feedback row on an event
    linked to more than one category increments the count in each of that
    event's category buckets -- this double-counts by design, mirroring
    the event's real category membership.
    """
    context_items_result = await session.execute(
        select(ContextItem)
        .where(ContextItem.watch_id == watch.id)
        .order_by(ContextItem.created_at, ContextItem.id)
    )
    context_items = list(context_items_result.scalars())
    system_context = [item for item in context_items if item.kind == "system"]
    user_context = [item for item in context_items if item.kind == "user"]

    relation_result = await session.execute(
        select(EventRelation).where(
            or_(
                EventRelation.from_event_id == event.id,
                EventRelation.to_event_id == event.id,
            )
        )
    )
    matched_via: dict[uuid.UUID, set[str]] = {}
    for row in relation_result.scalars():
        other_event_id = row.to_event_id if row.from_event_id == event.id else row.from_event_id
        matched_via.setdefault(other_event_id, set()).add(f"relation:{row.relation}")

    distances: dict[uuid.UUID, float] = {}
    for candidate in await find_nearest_events(session, event, limit=vector_limit):
        matched_via.setdefault(candidate.event_id, set()).add("vector")
        distances[candidate.event_id] = candidate.distance

    related_events_by_id: dict[uuid.UUID, Event] = {}
    if matched_via:
        events_result = await session.execute(
            select(Event).where(Event.id.in_(matched_via.keys()))
        )
        related_events_by_id = {row.id: row for row in events_result.scalars()}

    own_entities = set(event.entities)
    related_events = [
        RelatedEvent(
            event=related_event,
            matched_via=frozenset(tags),
            distance=distances.get(related_event_id),
            shared_entities=sorted(own_entities & set(related_event.entities)),
        )
        for related_event_id, tags in matched_via.items()
        if (related_event := related_events_by_id.get(related_event_id)) is not None
    ]
    # Deterministic order: vector-nearest first (ascending distance),
    # relation-only events (no distance) after, tie-broken by event id.
    related_events.sort(
        key=lambda re: (
            re.distance if re.distance is not None else float("inf"),
            re.event.id,
        )
    )

    cutoff = datetime.now(UTC) - timedelta(days=feedback_window_days)
    feedback_result = await session.execute(
        select(Category.slug, Feedback.verdict, func.count().label("count"))
        .select_from(Feedback)
        .join(EventCategory, EventCategory.event_id == Feedback.event_id)
        .join(Category, Category.id == EventCategory.category_id)
        .where(Feedback.watch_id == watch.id, Feedback.created_at >= cutoff)
        .group_by(Category.slug, Feedback.verdict)
        .order_by(Category.slug, Feedback.verdict)
    )
    feedback_summary = [
        FeedbackBucket(
            category_slug=row.slug,
            verdict=row.verdict,
            count=row.count,  # type: ignore[arg-type]
        )
        for row in feedback_result
    ]

    return ContextBundle(
        system_context=system_context,
        user_context=user_context,
        related_events=related_events,
        feedback_summary=feedback_summary,
    )
