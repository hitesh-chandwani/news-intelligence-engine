"""Historical relation stage (#29, `design.md` §5 stage 9, §8 FR-018).

`relate_stage` is a `STAGE_REGISTRY` `StageFn` (per #18): for every
scored `event` row not yet related, finds vector-nearest
(`find_nearest_events`, #27) and entity-overlapping
(`find_entity_overlapping_events`, #29's own addition to `match.py`)
historical events from the same watch, asks the LLM which pairs are
actually related and how, and persists the result as `event_relation`
(#10) rows.

Relations this stage creates are always directional **from the event
being related, to the historical candidate**
(`from_event_id = event.id`, `to_event_id = candidate.event_id`).

Replaces the `("relate", _not_yet_implemented)` `STAGE_REGISTRY` entry
from #18.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.llm.client import LLMClient
from nie.models import Event, EventRelation
from nie.pipeline.match import find_entity_overlapping_events, find_nearest_events

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "relate.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

_NO_CANDIDATES_TEXT = "(no candidate events found)"

# A separate, independently tunable constant from `match.py`'s
# `CONTEXT_EVENT_LIMIT`/`MATCH_CANDIDATE_LIMIT` -- different call site
# (this stage's own vector-nearest lookup), same "not a `Settings`/
# `.env` field" precedent as those two constants.
RELATE_VECTOR_LIMIT = 5


class ProposedRelation(BaseModel):
    """One LLM-proposed relation between the event being related and a
    candidate historical event, per `relate.md`.

    There is no `model_validator` here for candidate-membership or
    self-relation checks (unlike `AdjudicationResult`, #25) -- this model
    has no access to the event-specific candidate set or `event.id` at
    parse time. Those checks are `relate_stage`'s own job, applied
    per-relation rather than failing the whole response.
    """

    event_id: uuid.UUID
    relation: Literal["precedes", "similar", "escalation-of", "context-for"]
    rationale: str = Field(min_length=1)


class RelationSet(BaseModel):
    """The LLM's full structured response for one `event` row: zero, one,
    or many proposed relations. An empty `relations` list is a valid,
    normal response -- not an error."""

    relations: list[ProposedRelation]


class _Candidate(NamedTuple):
    """One merged, deduped candidate historical event for `event`, tagged
    with which source(s) found it -- internal to this module, mirroring
    `score.py`'s `RelatedEvent`."""

    event: Event
    matched_via: frozenset[str]
    distance: float | None
    shared_entities: list[str]


async def _find_candidates(session: AsyncSession, event: Event) -> list[_Candidate]:
    """Merge/dedupe `find_nearest_events` (tag `"vector"`) and
    `find_entity_overlapping_events` (tag `"entity_overlap"`) by
    `event_id` into a `matched_via: dict[uuid.UUID, set[str]]` map --
    exactly like `build_context_bundle`'s existing merge pattern
    (`score.py`, #27): an id found by both sources appears once, tags
    unioned. Then loads the full `Event` rows for the merged id set.

    `shared_entities` is computed uniformly for every merged candidate
    (not only ones found via entity overlap), same as
    `build_context_bundle`'s own `shared_entities` -- the sorted
    intersection of `event.entities` and the candidate's `entities`.

    Ordered by vector distance ascending first (candidates with no
    distance, i.e. entity-overlap-only, sort after), `Event.id` as the
    tie-break -- same ordering precedent as `build_context_bundle`'s
    `related_events.sort`. Returns `[]` when neither source finds
    anything (e.g. `event` is the only row for its watch).
    """
    matched_via: dict[uuid.UUID, set[str]] = {}
    distances: dict[uuid.UUID, float] = {}

    for candidate in await find_nearest_events(session, event, limit=RELATE_VECTOR_LIMIT):
        matched_via.setdefault(candidate.event_id, set()).add("vector")
        distances[candidate.event_id] = candidate.distance

    for overlap_candidate in await find_entity_overlapping_events(session, event):
        matched_via.setdefault(overlap_candidate.event_id, set()).add("entity_overlap")

    if not matched_via:
        return []

    events_result = await session.execute(select(Event).where(Event.id.in_(matched_via.keys())))
    events_by_id = {row.id: row for row in events_result.scalars()}

    own_entities = set(event.entities)
    candidates = [
        _Candidate(
            event=candidate_event,
            matched_via=frozenset(tags),
            distance=distances.get(candidate_id),
            shared_entities=sorted(own_entities & set(candidate_event.entities)),
        )
        for candidate_id, tags in matched_via.items()
        if (candidate_event := events_by_id.get(candidate_id)) is not None
    ]
    candidates.sort(
        key=lambda candidate: (
            candidate.distance if candidate.distance is not None else float("inf"),
            candidate.event.id,
        )
    )
    return candidates


def _format_candidates(candidates: list[_Candidate]) -> str:
    """Render the merged candidate list, in `_find_candidates`'s own
    order, or the explicit "no candidate events found" case for an empty
    list -- same rendering convention as `score.py`'s
    `_format_related_events`."""
    if not candidates:
        return _NO_CANDIDATES_TEXT

    lines = []
    for candidate in candidates:
        event = candidate.event
        matched_via = ", ".join(sorted(candidate.matched_via))
        shared_entities = (
            ", ".join(candidate.shared_entities) if candidate.shared_entities else "(none)"
        )
        lines.append(
            f"- event_id: {event.id}\n"
            f"  title: {event.title}\n"
            f"  fact_summary: {event.fact_summary}\n"
            f"  event_date: {event.event_date if event.event_date is not None else 'unknown'}\n"
            f"  matched_via: {matched_via}\n"
            f"  shared_entities: {shared_entities}"
        )
    return "\n".join(lines)


def _build_messages(event: Event, candidates: list[_Candidate]) -> list[dict[str, str]]:
    """Render `relate.md`'s candidate list first, then the event being
    related last -- same ordering rationale as `score.md`'s
    `_build_messages`."""
    prompt = _PROMPT_TEMPLATE.format(
        candidates=_format_candidates(candidates),
        title=event.title,
        fact_summary=event.fact_summary,
        interpretation=event.interpretation,
        entities=event.entities,
    )
    return [{"role": "user", "content": prompt}]


async def relate_stage(session: AsyncSession, *, client: LLMClient | None = None) -> dict[str, int]:
    """Ask the LLM which historical candidates every scored, not-yet-related
    `event` row actually relates to, and persist the surviving proposals as
    `event_relation` rows.

    `client` defaults to constructing its own `LLMClient()` when not given,
    same as `adjudicate_stage`/`score_stage`.

    Selects rows via `select(Event).where(Event.relevance.isnot(None),
    Event.related_at.is_(None))` and processes them sequentially (no
    concurrency), same style as `adjudicate_stage`/`score_stage`.
    `relevance IS NOT NULL` means "scored" (no special-casing by value,
    same "not this stage's job to gate on `irrelevant`" precedent
    `score_stage` set). `related_at IS NULL` means this event has never
    had a successfully-parsed `RelationSet` response recorded for it by a
    prior call -- once `related_at` is set, an event is excluded from
    every future selection *regardless of how many relations it ended up
    with*, including zero (#47; this replaces the older "no existing
    outbound `event_relation` row" criterion, which reselected
    zero-relation events forever).

    For each row: calls `_find_candidates` (merges `find_nearest_events` +
    `find_entity_overlapping_events`, tags, loads full `Event` rows),
    builds `messages` from `relate.md`, and calls
    `client.call_structured(messages, RelationSet)`.

    A `json.JSONDecodeError`/`pydantic.ValidationError` still raised after
    `call_structured`'s own internal validate-then-retry-once is a
    whole-event failure: the event's `event_relation` rows are left
    completely unchanged (none added), `related_at` is left unset (same
    "leave the row unmodified, retry next run" precedent #45/#46 set for
    `extract_attempts`/`score_attempts`, though this stage has no
    attempt cap yet -- see the issue's "Out of scope"), and it is counted
    under `"skipped"`; the loop continues to the next event. Any other
    exception propagates out of `relate_stage` uncaught, same precedent
    as `score_stage`/`adjudicate_stage`.

    On a successful, structurally-valid `RelationSet`, `event.related_at`
    is set to `datetime.now(UTC)` regardless of how many relations survive
    filtering below -- including zero, whether because the LLM proposed
    none at all or because every proposal was dropped as
    hallucinated/self-relating -- mirroring `Source.extracted_at`'s "set
    once on a successful pass" semantics (#47). Each `ProposedRelation` is
    then filtered independently (an event's other, valid proposals are
    never discarded because one entry in the same response is bad) --
    none of these three checks fail the whole event or count it as
    `"skipped"`:

    - Dropped if `event_id` is not in this event's merged candidate id
      set (hallucinated id) -- mirrors `adjudicate_stage`'s "not in
      `candidate_ids` -> skip" check.
    - Dropped if `event_id == event.id` (self-relation; would violate
      `ck_event_relation_no_self_relation`) -- defense in depth only,
      since both candidate-finding functions already exclude `event`
      itself, so this is already caught by the check above, but is
      checked (and regression-tested) explicitly rather than relying on
      that exclusion silently holding.
    - Dropped if `(to_event_id, relation)` duplicates a pair already
      persisted for this `from_event_id`, or already inserted earlier in
      this same response -- this is what prevents the composite primary
      key `(from_event_id, to_event_id, relation)` from ever being
      violated; there is no `try`/`except` around the insert.

    Does not call `session.commit()` -- the runner (#18) commits after
    the stage returns.

    Returns `{"related": N, "skipped": S}` -- `related` counts events for
    which at least one `event_relation` row was persisted this call. An
    event whose only proposals were all filtered out, or whose
    `RelationSet` was legitimately empty, counts toward neither
    `"related"` nor `"skipped"` -- it simply produced zero rows this pass
    (not an error; `related_at` is still set for it, per #47, so it is
    not reselected on a future call).
    """
    if client is None:
        client = LLMClient()

    result = await session.execute(
        select(Event).where(
            Event.relevance.isnot(None),
            Event.related_at.is_(None),
        )
    )
    events = result.scalars().all()

    related_count = 0
    skipped_count = 0

    for event in events:
        candidates = await _find_candidates(session, event)
        candidate_ids = {candidate.event.id for candidate in candidates}
        messages = _build_messages(event, candidates)

        try:
            relation_set = await client.call_structured(messages, RelationSet)
        except (json.JSONDecodeError, ValidationError):
            skipped_count += 1
            continue

        assert isinstance(relation_set, RelationSet)

        # Set once per successful, structurally-valid parse -- regardless
        # of how many relations survive filtering below, including zero
        # -- so this event is never reselected by a future call (#47).
        event.related_at = datetime.now(UTC)

        existing_result = await session.execute(
            select(EventRelation.to_event_id, EventRelation.relation).where(
                EventRelation.from_event_id == event.id
            )
        )
        persisted_pairs = {(row.to_event_id, row.relation) for row in existing_result}
        seen_pairs: set[tuple[uuid.UUID, str]] = set()

        inserted_any = False
        for proposed in relation_set.relations:
            if proposed.event_id not in candidate_ids:
                continue
            if proposed.event_id == event.id:
                continue

            pair = (proposed.event_id, proposed.relation)
            if pair in persisted_pairs or pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            session.add(
                EventRelation(
                    from_event_id=event.id,
                    to_event_id=proposed.event_id,
                    relation=proposed.relation,
                    rationale=proposed.rationale,
                )
            )
            inserted_any = True

        if inserted_any:
            related_count += 1

    return {"related": related_count, "skipped": skipped_count}
