"""Vector match stage (#22, `design.md` §5 stage 5, §8 "Dedup & historical
linking").

`find_candidate_events` is a plain async function, **not** a
`STAGE_REGISTRY` `StageFn` -- see the issue's "Resolving 'is `match` a
`STAGE_REGISTRY` stage?'" section. Its real output (a per-source ranked
candidate list) isn't a count and is only meaningful as adjudicate
(#25)'s immediate input, so it doesn't decompose into an independent
full-table batch pass the way `discover_stage`/`extract_stage`/
`embed_stage` do. #25 will call this function directly for each source
it processes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.config import Settings
from nie.models import Event, Source

# Not a `Settings`/`.env` field -- `design.md` §14's config table has no
# candidate-limit entry, same "not a config field" precedent as
# `discover.py`'s `STUB_FIXTURES_DIR`. `limit` stays a keyword argument
# (default this constant) so callers/tests can override it without
# touching the constant itself.
MATCH_CANDIDATE_LIMIT = 5

# A separate, independently tunable constant for #27's historical-context
# lookup (`find_nearest_events`) -- different call site than
# `MATCH_CANDIDATE_LIMIT`, and, per `design.md` §14, not a `Settings`/
# `.env` field either.
CONTEXT_EVENT_LIMIT = 5


class MatchCandidate(NamedTuple):
    event_id: uuid.UUID
    distance: float


async def find_candidate_events(
    session: AsyncSession,
    source: Source,
    *,
    limit: int = MATCH_CANDIDATE_LIMIT,
) -> list[MatchCandidate]:
    """Rank same-watch, in-window `event` rows by cosine distance to
    `source.embedding`, nearest first.

    Raises `ValueError` if `source.embedding is None` -- querying pgvector
    with a null vector is a caller bug (this function's contract requires
    an already-embedded source, #21's `embed_stage` output), not an empty
    result, so no query is issued in that case.

    Reads `Settings().dedup_window_days` fresh inside this function, same
    pattern `discover_stage` uses for its own `settings = Settings()`
    call, rather than taking it as a parameter.

    An event qualifies when its watch matches `source.watch_id` and
    `COALESCE(event.event_date, event.discovered_at) >= cutoff` --
    `Event.event_date` is nullable while `Event.discovered_at` is not, so
    `func.coalesce(...)` is required or every event with a null
    `event_date` would silently be excluded from the `>=` comparison
    (`NULL >= cutoff` is never true in SQL).

    Returns at most `limit` candidates ordered by cosine distance
    ascending, with `Event.id` as a secondary sort key purely for
    deterministic output when two candidates tie on distance (no product
    meaning). Returns `[]` when nothing qualifies.
    """
    if source.embedding is None:
        raise ValueError("source.embedding is None; cannot run a vector match query")

    settings = Settings()
    cutoff = datetime.now(UTC) - timedelta(days=settings.dedup_window_days)

    distance = Event.embedding.cosine_distance(source.embedding)
    query = (
        select(Event.id, distance.label("distance"))
        .where(
            Event.watch_id == source.watch_id,
            func.coalesce(Event.event_date, Event.discovered_at) >= cutoff,
        )
        .order_by(distance, Event.id)
        .limit(limit)
    )
    result = await session.execute(query)
    return [MatchCandidate(event_id=row.id, distance=row.distance) for row in result]


async def find_nearest_events(
    session: AsyncSession,
    event: Event,
    *,
    limit: int = CONTEXT_EVENT_LIMIT,
) -> list[MatchCandidate]:
    """Rank same-watch `event` rows (excluding `event` itself) by cosine
    distance to `event.embedding`, nearest first.

    Sibling to `find_candidate_events`, built for #27's historical-context
    bundle rather than #25's dedup/adjudication candidate list -- see the
    issue's "Can `find_candidate_events` be reused as-is? No." section.
    Unlike `find_candidate_events`, there is **no** recency filter here:
    stage 8's context bundle wants relevant history regardless of age, not
    only same-run dedup candidates.

    Returns at most `limit` results ordered by cosine distance ascending,
    with `Event.id` as a secondary sort key for deterministic output on a
    tie, same precedent as `find_candidate_events`. Returns `[]` when
    nothing else qualifies (e.g. `event` is the only row for its watch).
    Does not alter `find_candidate_events`'s existing behavior.
    """
    distance = Event.embedding.cosine_distance(event.embedding)
    query = (
        select(Event.id, distance.label("distance"))
        .where(Event.watch_id == event.watch_id, Event.id != event.id)
        .order_by(distance, Event.id)
        .limit(limit)
    )
    result = await session.execute(query)
    return [MatchCandidate(event_id=row.id, distance=row.distance) for row in result]
