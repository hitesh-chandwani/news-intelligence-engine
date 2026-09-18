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

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.config import Settings
from nie.llm.client import LLMClient
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

# Same "= 5" module-constant convention `MATCH_CANDIDATE_LIMIT`/
# `CONTEXT_EVENT_LIMIT`/`ENTITY_OVERLAP_LIMIT`/`RELATE_VECTOR_LIMIT`
# already use in this codebase. Not a `Settings`/`.env` field, same
# "stays a module-level constant" reasoning `FEEDBACK_WINDOW_DAYS` gives
# above -- #57's Out of scope explicitly keeps this off `Settings`.
FEEDBACK_NOTES_PER_BUCKET = 5

# Read-side prompt-rendering truncation only -- `note` itself stays an
# unconstrained `Text` column with no write-time cap (#50); this just
# keeps one verbose note from dominating the score-stage prompt.
FEEDBACK_NOTE_CHAR_LIMIT = 200


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
    # This bucket's `FEEDBACK_NOTES_PER_BUCKET` most recent non-blank
    # `Feedback.note` values (`created_at` descending), each truncated to
    # `FEEDBACK_NOTE_CHAR_LIMIT` chars with a trailing "..." marker when
    # truncated. Blank/null notes never appear here but still contribute
    # to `count` above. `[]` when this bucket has zero noted rows in the
    # feedback window -- see `_format_feedback_summary`.
    notes: list[str]


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

    Each bucket also carries up to `FEEDBACK_NOTES_PER_BUCKET` of that
    bucket's own non-blank `note` values (#50), most recent first by
    `created_at`, each truncated to `FEEDBACK_NOTE_CHAR_LIMIT` characters
    (#57). This reuses the same `watch_id`/`created_at >= cutoff` filter
    and `(category.slug, verdict)` grouping the count query above uses --
    a feedback row on a multi-category event surfaces its note in each of
    that event's category buckets, same double-counting-by-design as
    `count`. A withdrawn (`#51` hard-deleted) row's note stops appearing
    on the next call with no code change here, since this query reads
    live table state, same as the count query above.
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
    # Same `watch_id`/`created_at >= cutoff` filter and `(category.slug,
    # verdict)` grouping as the count query above, just over individual
    # non-blank-noted rows instead of an aggregate -- not a separate
    # filter surface. Ordered `created_at` descending overall, which
    # preserves each bucket's own most-recent-first order as the loop
    # below buckets rows by key and caps each bucket at
    # `FEEDBACK_NOTES_PER_BUCKET`; the `Feedback.id` tie-break only
    # matters for rows sharing one `created_at` value.
    notes_result = await session.execute(
        select(Category.slug, Feedback.verdict, Feedback.note)
        .select_from(Feedback)
        .join(EventCategory, EventCategory.event_id == Feedback.event_id)
        .join(Category, Category.id == EventCategory.category_id)
        .where(
            Feedback.watch_id == watch.id,
            Feedback.created_at >= cutoff,
            Feedback.note.isnot(None),
        )
        .order_by(Feedback.created_at.desc(), Feedback.id.desc())
    )
    bucket_notes: dict[tuple[str, str], list[str]] = {}
    for note_row in notes_result:
        key = (note_row.slug, note_row.verdict)
        notes = bucket_notes.setdefault(key, [])
        if len(notes) < FEEDBACK_NOTES_PER_BUCKET:
            note_text = note_row.note
            assert note_text is not None  # excluded by `Feedback.note.isnot(None)` above
            if len(note_text) > FEEDBACK_NOTE_CHAR_LIMIT:
                note_text = note_text[:FEEDBACK_NOTE_CHAR_LIMIT] + "..."
            notes.append(note_text)

    feedback_summary = [
        FeedbackBucket(
            category_slug=row.slug,
            verdict=row.verdict,
            count=row.count,  # type: ignore[arg-type]
            notes=bucket_notes.get((row.slug, row.verdict), []),
        )
        for row in feedback_result
    ]

    return ContextBundle(
        system_context=system_context,
        user_context=user_context,
        related_events=related_events,
        feedback_summary=feedback_summary,
    )


# ---------------------------------------------------------------------------
# score_stage (#28, `design.md` §5 stage 8)
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "score.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

_NO_CONTEXT_TEXT = "(none)"
_NO_RELATED_EVENTS_TEXT = "(no related events found)"
_NO_FEEDBACK_TEXT = "(no feedback recorded yet)"


class ScoreResult(BaseModel):
    """The LLM's structured scoring verdict for one `event` row, per
    `score.md`. Field names/values mirror `Event`'s `CheckConstraint`s in
    `models.py` exactly (same enum values; `direction`/`reason`/
    `confidence` here carry the `impact_` prefix `Event`'s own columns do).
    """

    relevance: Literal["irrelevant", "low", "medium", "high"]
    importance: Literal["low", "medium", "high", "critical"]
    impact_direction: Literal["bullish", "bearish", "neutral", "unclear"]
    impact_reason: str = Field(min_length=1)
    impact_confidence: Literal["low", "medium", "high"]


def _format_context_items(items: list[ContextItem]) -> str:
    """Render a `system_context`/`user_context` list as one `label: body`
    line per item, or the explicit "(none)" case for an empty list."""
    if not items:
        return _NO_CONTEXT_TEXT
    return "\n".join(f"- {item.label}: {item.body}" for item in items)


def _format_related_events(related_events: list[RelatedEvent]) -> str:
    """Render the related-events section, in the bundle's own order
    (nearest/most-linked first), or the explicit "no related events found"
    case for an empty list."""
    if not related_events:
        return _NO_RELATED_EVENTS_TEXT

    lines = []
    for related in related_events:
        event = related.event
        matched_via = ", ".join(sorted(related.matched_via))
        shared_entities = (
            ", ".join(related.shared_entities) if related.shared_entities else "(none)"
        )
        lines.append(
            f"- event_id: {event.id}\n"
            f"  title: {event.title}\n"
            f"  fact_summary: {event.fact_summary}\n"
            f"  interpretation: {event.interpretation}\n"
            f"  event_date: {event.event_date if event.event_date is not None else 'unknown'}\n"
            f"  matched_via: {matched_via}\n"
            f"  shared_entities: {shared_entities}"
        )
    return "\n".join(lines)


def _format_feedback_summary(feedback_summary: list[FeedbackBucket]) -> str:
    """Render one `"<category_slug>: <count> x '<verdict>'"` line per
    bucket, or the explicit "no feedback recorded yet" case for an empty
    list.

    A bucket with any notes (#57) gets an indented `  notes: "...", "..."`
    sub-line directly under its count line, listing that bucket's
    (already recency-capped, char-capped) `notes` sample. A bucket with
    zero noted rows keeps its count-only line exactly as before -- no
    sub-line at all, not an empty one.
    """
    if not feedback_summary:
        return _NO_FEEDBACK_TEXT
    lines = []
    for bucket in feedback_summary:
        lines.append(f"{bucket.category_slug}: {bucket.count} x '{bucket.verdict}'")
        if bucket.notes:
            quoted_notes = ", ".join(f'"{note}"' for note in bucket.notes)
            lines.append(f"  notes: {quoted_notes}")
    return "\n".join(lines)


def _build_messages(event: Event, bundle: ContextBundle) -> list[dict[str, str]]:
    """Render `score.md`'s five ordered sections: silver background
    context, user context, related historical events, feedback summary,
    then the event itself last -- so the model reasons over context first
    and judges the specific event last."""
    prompt = _PROMPT_TEMPLATE.format(
        system_context=_format_context_items(bundle.system_context),
        user_context=_format_context_items(bundle.user_context),
        related_events=_format_related_events(bundle.related_events),
        feedback_summary=_format_feedback_summary(bundle.feedback_summary),
        title=event.title,
        fact_summary=event.fact_summary,
        interpretation=event.interpretation,
        entities=event.entities,
    )
    return [{"role": "user", "content": prompt}]


async def score_stage(session: AsyncSession, *, client: LLMClient | None = None) -> dict[str, int]:
    """Score every `event` row not yet scored: `relevance`, `importance`,
    `impact_direction`, `impact_reason`, `impact_confidence`.

    `client` defaults to constructing its own `LLMClient()` when not given,
    same as `adjudicate_stage`/`synthesize_stage`. Reads `Settings().
    max_score_attempts` fresh on every call, same "construct `Settings()`
    inside the stage" pattern `extract_stage`/`discover_stage` use.

    Selects rows via
    `select(Event).where(Event.relevance.is_(None), Event.score_attempts <
    settings.max_score_attempts)` and processes them sequentially (no
    concurrency), same style as `adjudicate_stage`/`synthesize_stage`.
    `relevance` is the one field of the five this stage always sets
    regardless of verdict -- including `irrelevant` -- so "has this event
    been scored yet" is exactly `relevance IS NULL`. An event already at
    `score_attempts >= max_score_attempts` is excluded by this clause
    entirely (#46): it's never loaded, never passed to `call_structured`,
    and its `score_attempts` is left unchanged.

    For each selected row: loads its `Watch` (`Event.watch_id`), calls
    `build_context_bundle(session, watch, event)` (#27), builds `messages`
    from `score.md` (event + full bundle), increments `event.score_attempts`
    by 1 (mirroring `extract_stage`'s `source.extract_attempts += 1`
    placement -- before the call that might fail; unlike extract, score
    has no "content already supplied" skip case, so every selected row
    gets exactly one increment per call), and calls
    `client.call_structured(messages, ScoreResult)`.

    On a successful response, writes all five fields onto the `Event` row
    from `ScoreResult`, uniformly -- no special-casing `irrelevant` (it
    still gets `importance`/`impact_*` filled in from the same response;
    the notify gate, #30, is what short-circuits on `irrelevant`, not this
    stage). `fact_summary`/`interpretation` are read-only here, rendered
    into the prompt only -- never written.

    A `json.JSONDecodeError`/`pydantic.ValidationError` still raised after
    `call_structured`'s own internal validate-then-retry-once is a per-row
    failure: the event row is left with its five score fields unmodified
    (all stay `NULL`, `score_attempts` already incremented above) and
    counted under `"skipped"`; the loop continues to the next event. Any
    other exception propagates out of `score_stage` uncaught, same
    precedent as `adjudicate_stage`/`synthesize_stage`.

    Does not call `session.commit()` -- the runner (#18) commits after the
    stage returns.

    Returns `{"irrelevant": N, "scored": M, "skipped": S, "score_capped":
    K}` -- `irrelevant` counts `relevance == "irrelevant"` responses,
    `scored` counts every other successfully-persisted `relevance` value
    combined (`low`/`medium`/`high`), `skipped` counts validation failures
    this call, and `score_capped` counts `relevance IS NULL` rows excluded
    by the cap this call (a separate count query -- `Event.relevance.
    is_(None), Event.score_attempts >= settings.max_score_attempts` --
    not folded into the main loop, same pattern `extract_stage` uses for
    `extract_capped`).
    """
    if client is None:
        client = LLMClient()
    settings = Settings()

    result = await session.execute(
        select(Event).where(
            Event.relevance.is_(None), Event.score_attempts < settings.max_score_attempts
        )
    )
    events = result.scalars().all()

    capped_result = await session.execute(
        select(Event.id).where(
            Event.relevance.is_(None), Event.score_attempts >= settings.max_score_attempts
        )
    )
    score_capped_count = len(capped_result.scalars().all())

    irrelevant_count = 0
    scored_count = 0
    skipped_count = 0

    for event in events:
        watch_result = await session.execute(select(Watch).where(Watch.id == event.watch_id))
        watch = watch_result.scalar_one()

        bundle = await build_context_bundle(session, watch, event)
        messages = _build_messages(event, bundle)

        event.score_attempts += 1
        try:
            score = await client.call_structured(messages, ScoreResult)
        except (json.JSONDecodeError, ValidationError):
            skipped_count += 1
            continue

        assert isinstance(score, ScoreResult)

        event.relevance = score.relevance
        event.importance = score.importance
        event.impact_direction = score.impact_direction
        event.impact_reason = score.impact_reason
        event.impact_confidence = score.impact_confidence

        if score.relevance == "irrelevant":
            irrelevant_count += 1
        else:
            scored_count += 1

    return {
        "irrelevant": irrelevant_count,
        "scored": scored_count,
        "skipped": skipped_count,
        "score_capped": score_capped_count,
    }
