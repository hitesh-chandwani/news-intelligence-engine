"""Adjudicate stage (#25, `design.md` §5 stage 6).

`adjudicate_stage` is a `STAGE_REGISTRY` `StageFn` (per #18): for every
triaged-in and embedded `source` row it calls `find_candidate_events`
(#22) to get a ranked list of same-watch, in-window candidate `event`
rows, asks the LLM to decide `new` / `existing(event_id)` / `noise` plus a
`materiality` level, and persists that decision on the `source` row
itself so #26 (synthesize) knows exactly what to do with it. This stage
does **not** build the event record or link `event_source` -- that stays
#26's job (see the issue's "Resolving..." section for why `adjudicate` is
its own stage rather than merged into `synthesize`).

Unlike `match` (#22), `adjudicate` *is* a `STAGE_REGISTRY` stage: for
every qualifying `source` row, independently, the answer is one
structured verdict with a side effect on that row alone -- the same
"select a status batch, loop, mutate each row, return counts" shape as
`triage_stage`/`embed_stage`. It calls `find_candidate_events` (a plain
function, not a registered stage) once per source it processes.

Replaces the `("adjudicate", _not_yet_implemented)` `STAGE_REGISTRY`
entry from #18.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.llm.client import LLMClient
from nie.models import Event, Source
from nie.pipeline.match import MatchCandidate, find_candidate_events

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "adjudicate.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

_NO_CANDIDATES_TEXT = "(no candidate events found)"


class AdjudicationResult(BaseModel):
    """The LLM's structured verdict for one `source` row, per `adjudicate.md`.

    A `model_validator` enforces the decision/materiality/event_id pairing
    so a structurally-valid-but-nonsensical response fails validation (and
    hits `call_structured`'s existing validate-then-retry-once path, same
    as any other malformed response) rather than being silently persisted:

    - `decision == "existing"` iff `event_id is not None`.
    - `decision == "new"` requires `materiality == "material"` (a
      brand-new event is definitionally a material development).
    - `decision == "noise"` requires `materiality == "none"`.
    - (`existing` accepts any of `none` / `minor` / `material`.)

    `event_id` *membership* in that source's own candidate list is **not**
    checked here -- this model has no access to that source-specific
    candidate set. That check is `adjudicate_stage`'s own job.
    """

    decision: Literal["new", "existing", "noise"]
    event_id: uuid.UUID | None = None
    materiality: Literal["none", "minor", "material"]

    @model_validator(mode="after")
    def _check_pairing(self) -> AdjudicationResult:
        if self.decision == "existing" and self.event_id is None:
            raise ValueError("decision == 'existing' requires a non-null event_id")
        if self.decision != "existing" and self.event_id is not None:
            raise ValueError("event_id must be null unless decision == 'existing'")
        if self.decision == "new" and self.materiality != "material":
            raise ValueError("decision == 'new' requires materiality == 'material'")
        if self.decision == "noise" and self.materiality != "none":
            raise ValueError("decision == 'noise' requires materiality == 'none'")
        return self


def _format_candidates(
    events_by_id: dict[uuid.UUID, Event], candidates: list[MatchCandidate]
) -> str:
    """Render the candidate section, nearest-first, or the explicit
    "no candidates" case for an empty list."""
    if not candidates:
        return _NO_CANDIDATES_TEXT

    lines = []
    for candidate in candidates:
        event = events_by_id[candidate.event_id]
        lines.append(
            f"- event_id: {event.id}\n"
            f"  title: {event.title}\n"
            f"  fact_summary: {event.fact_summary}\n"
            f"  event_date: {event.event_date if event.event_date is not None else 'unknown'}"
        )
    return "\n".join(lines)


async def _load_candidate_events(
    session: AsyncSession, candidates: list[MatchCandidate]
) -> dict[uuid.UUID, Event]:
    """Load the full `Event` rows (title/fact_summary/event_date) for the
    candidate ids returned by `find_candidate_events` -- the prompt must
    give the LLM actual event content to compare against, not bare
    `event_id`/distance pairs."""
    if not candidates:
        return {}
    ids = [candidate.event_id for candidate in candidates]
    result = await session.execute(select(Event).where(Event.id.in_(ids)))
    return {event.id: event for event in result.scalars().all()}


def _build_messages(
    source: Source, events_by_id: dict[uuid.UUID, Event], candidates: list[MatchCandidate]
) -> list[dict[str, str]]:
    prompt = _PROMPT_TEMPLATE.format(
        title=source.title,
        content=source.content,
        candidates=_format_candidates(events_by_id, candidates),
    )
    return [{"role": "user", "content": prompt}]


async def adjudicate_stage(
    session: AsyncSession, *, client: LLMClient | None = None
) -> dict[str, int]:
    """Ask the LLM for a new/existing/noise decision on every triaged-in,
    embedded `source` row.

    `client` defaults to constructing its own `LLMClient()` when not
    given, same as `triage_stage`.

    Selects rows via `select(Source).where(Source.status == "extracted",
    Source.embedding.isnot(None))` and processes them sequentially (no
    concurrency), same style as `extract_stage`/`triage_stage`. A row
    still `"extracted"` with no embedding, or already `"triaged_out"` /
    `"extract_failed"` / `"adjudicated"` / `"processed"`, is never
    selected.

    For each row: calls `find_candidate_events(session, source)` (#22),
    loads the full `Event` rows for the returned candidate ids, builds
    `messages` from `adjudicate.md`, and calls
    `client.call_structured(messages, AdjudicationResult)`.

    `AdjudicationResult`'s own `model_validator` enforces the decision/
    materiality/event_id pairing rules. Additionally, a `decision ==
    "existing"` result whose `event_id` is not one of the candidate
    `event_id`s just passed to the LLM for this source (including the
    empty-candidate-list case) is treated exactly like a validation
    failure: the row is left unmodified and counted under `"skipped"`.

    Any other `json.JSONDecodeError` / `pydantic.ValidationError` raised
    by `call_structured` (still malformed after its own internal
    validate-then-retry-once) is likewise a per-row failure: the row is
    left completely unmodified and the loop continues to the next row --
    matching `triage_stage`'s/`extract_stage`'s per-row isolation. Any
    other exception propagates out of `adjudicate_stage` uncaught, same
    precedent as `extract_stage`/`triage_stage`; the runner's own
    try/except records `stats["adjudicate"] = {"error": ...}`.

    Persistence:
    - `noise` -> `status = "processed"`, `adjudication_decision =
      "noise"`, `adjudication_materiality = "none"`,
      `adjudication_event_id = NULL`.
    - `new` -> `status = "adjudicated"`, `adjudication_decision = "new"`,
      `adjudication_materiality = "material"`, `adjudication_event_id =
      NULL`.
    - `existing` -> `status = "adjudicated"`, `adjudication_decision =
      "existing"`, `adjudication_event_id` = the LLM's `event_id`,
      `adjudication_materiality` = the LLM's materiality.

    Does not write any `event_source` row -- that stays #26's job. Does
    not call `session.commit()` -- #18's runner commits after the stage
    returns.

    Returns `{"new": N, "existing": M, "noise": K, "skipped": S}`.
    """
    if client is None:
        client = LLMClient()

    result = await session.execute(
        select(Source).where(Source.status == "extracted", Source.embedding.isnot(None))
    )
    sources = result.scalars().all()

    new_count = 0
    existing_count = 0
    noise_count = 0
    skipped_count = 0

    for source in sources:
        candidates = await find_candidate_events(session, source)
        candidate_ids = {candidate.event_id for candidate in candidates}
        events_by_id = await _load_candidate_events(session, candidates)
        messages = _build_messages(source, events_by_id, candidates)

        try:
            verdict = await client.call_structured(messages, AdjudicationResult)
        except (json.JSONDecodeError, ValidationError):
            skipped_count += 1
            continue

        assert isinstance(verdict, AdjudicationResult)

        if verdict.decision == "existing" and verdict.event_id not in candidate_ids:
            skipped_count += 1
            continue

        if verdict.decision == "noise":
            source.status = "processed"
            source.adjudication_decision = "noise"
            source.adjudication_materiality = "none"
            source.adjudication_event_id = None
            noise_count += 1
        elif verdict.decision == "new":
            source.status = "adjudicated"
            source.adjudication_decision = "new"
            source.adjudication_materiality = "material"
            source.adjudication_event_id = None
            new_count += 1
        else:
            source.status = "adjudicated"
            source.adjudication_decision = "existing"
            source.adjudication_event_id = verdict.event_id
            source.adjudication_materiality = verdict.materiality
            existing_count += 1

    return {
        "new": new_count,
        "existing": existing_count,
        "noise": noise_count,
        "skipped": skipped_count,
    }
