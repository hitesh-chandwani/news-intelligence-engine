"""Synthesize stage (#26, `design.md` §5 stage 7).

`synthesize_stage` is a `STAGE_REGISTRY` `StageFn` (per #18): for every
`adjudicated` `source` row it turns that row's #25 adjudication decision
into a real `event` record change, always finishing by linking
`event_source` and advancing the source to `status = "processed"` -- the
terminal state for every source that reaches this stage. There are three
branches, keyed off `Source.adjudication_decision` /
`Source.adjudication_materiality`:

- `decision == "new"`: build a brand-new `Event` from an LLM-generated
  `EventRecord`, with a fresh `event_category` set and a freshly-computed
  embedding.
- `decision == "existing"` and `materiality == "material"`: load the
  existing `Event` at `Source.adjudication_event_id`, ask the LLM for an
  updated *full* `EventRecord` (title/fact_summary/interpretation/
  event_date/entities/categories, folding the new article in), overwrite
  the row, recompute its embedding, replace its `event_category` set, and
  bump `last_material_update_at`.
- `decision == "existing"` and `materiality in ("none", "minor")`: no LLM
  call, no record change -- just link `event_source` to the existing
  event.

All three branches insert exactly one `EventSource` row for this source.
`Category` rows are loaded once per `synthesize_stage` call (they don't
change mid-run), and a hallucinated/invalid slug is dropped leniently
rather than failing the whole row -- see the issue's "Resolving 'how are
categories validated?'" section. A response that's still malformed after
`call_structured`'s own internal retry is a per-row failure: the source is
left completely unmodified and counted under `"skipped"`.

Replaces the `("synthesize", _not_yet_implemented)` `STAGE_REGISTRY` entry
from #18.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.embeddings.fastembed import embed_text
from nie.llm.client import LLMClient
from nie.models import Category, Event, EventCategory, EventSource, Source

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "synthesize.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()

# Guaranteed to exist -- one of the 13 rows `nie.seed.categories.seed_categories`
# inserts. Used when every slug the LLM returned is invalid (or none were
# returned) so every event still gets at least one category (FR-011).
_FALLBACK_CATEGORY_SLUG = "other"

_EXISTING_CLAUSE = (
    ", and the existing event below that this article is a development of"
)

_EXISTING_EVENT_SECTION = """
This article is about an event this watch already knows about. Fold the
new article into an updated version of the *whole* record: merge its
entities with the existing ones, and keep the existing `event_date` unless
the new article clearly revises it.

Existing event:
  title: {event_title}
  fact_summary: {event_fact_summary}
  interpretation: {event_interpretation}
  event_date: {event_date}
  entities: {event_entities}
"""


class EventRecord(BaseModel):
    """The LLM's structured full-record response for the `new` branch and
    the `existing` + `material` branch alike -- design.md's "update the
    record" is a full-record refresh, not a diff/patch, so one response
    model covers both.

    `categories` is a list of `Category.slug` values -- validated/filtered
    against the current category table by `synthesize_stage`, not here
    (this model has no DB access): an unrecognized slug is dropped rather
    than failing validation, and an all-invalid/empty result falls back to
    `["other"]`. See the issue's "Resolving 'how are categories
    validated?'" section.
    """

    title: str = Field(min_length=1)
    fact_summary: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    event_date: datetime | None = None
    entities: list[str]
    categories: list[str] = Field(min_length=1)


def _filter_categories(slugs: list[str], valid_slugs: set[str]) -> list[str]:
    """Drop any slug not in `valid_slugs`, keeping the rest; fall back to
    `["other"]` if nothing survives (including an empty input list)."""
    filtered = [slug for slug in slugs if slug in valid_slugs]
    if not filtered:
        return [_FALLBACK_CATEGORY_SLUG]
    return filtered


def _build_messages(
    source: Source,
    categories_text: str,
    *,
    existing_event: Event | None = None,
) -> list[dict[str, str]]:
    """Render `synthesize.md`, including the existing-event context block
    only when `existing_event` is given (the `existing` + `material`
    branch); omitted entirely for the `new` branch."""
    if existing_event is not None:
        existing_clause = _EXISTING_CLAUSE
        existing_event_section = _EXISTING_EVENT_SECTION.format(
            event_title=existing_event.title,
            event_fact_summary=existing_event.fact_summary,
            event_interpretation=existing_event.interpretation,
            event_date=(
                existing_event.event_date
                if existing_event.event_date is not None
                else "unknown"
            ),
            event_entities=existing_event.entities,
        )
    else:
        existing_clause = ""
        existing_event_section = ""

    prompt = _PROMPT_TEMPLATE.format(
        title=source.title,
        content=source.content,
        categories=categories_text,
        existing_clause=existing_clause,
        existing_event_section=existing_event_section,
    )
    return [{"role": "user", "content": prompt}]


def _embed_event_text(title: str, fact_summary: str, interpretation: str) -> list[float]:
    """`f"{title}\\n\\n{fact_summary}\\n\\n{interpretation}"` -- extends
    `embed_stage`'s `title`/`content` convention by one field, per the
    issue's "Resolving 'how is the embedding computed?'" section."""
    return embed_text(f"{title}\n\n{fact_summary}\n\n{interpretation}")


async def synthesize_stage(
    session: AsyncSession, *, client: LLMClient | None = None
) -> dict[str, int]:
    """Turn every `adjudicated` `source` row's #25 decision into an event
    record change plus an `event_source` link, advancing it to
    `status = "processed"`.

    `client` defaults to constructing its own `LLMClient()` when not
    given, same as `adjudicate_stage`/`triage_stage`.

    Selects rows via `select(Source).where(Source.status == "adjudicated")`
    and processes them sequentially (no concurrency), same style as
    `adjudicate_stage`. Only `new`/`existing` decisions ever reach this
    status (`noise` is already terminal at `status = "processed"` from
    #25), so no further filtering on `adjudication_decision` is needed.

    `Category` rows are loaded once for the whole call, not once per
    source.

    Does not call `session.commit()` -- #18's runner commits after the
    stage returns. A `json.JSONDecodeError`/`pydantic.ValidationError`
    still raised after `call_structured`'s own internal retry is a
    per-row failure: the row is left completely unmodified and counted
    under `"skipped"`. Any other exception propagates out of
    `synthesize_stage` uncaught, same precedent as
    `extract_stage`/`triage_stage`/`adjudicate_stage`.

    Returns `{"new": N, "existing_updated": M, "existing_linked": K,
    "skipped": S}` (`existing_updated` = the `material` branch,
    `existing_linked` = the `none`/`minor` branch).
    """
    if client is None:
        client = LLMClient()

    category_result = await session.execute(select(Category))
    categories = category_result.scalars().all()
    valid_slugs = {category.slug for category in categories}
    category_id_by_slug = {category.slug: category.id for category in categories}
    categories_text = "\n".join(
        f"{category.slug}: {category.name}" for category in categories
    )

    result = await session.execute(select(Source).where(Source.status == "adjudicated"))
    sources = result.scalars().all()

    new_count = 0
    existing_updated_count = 0
    existing_linked_count = 0
    skipped_count = 0

    for source in sources:
        if source.adjudication_decision == "new":
            messages = _build_messages(source, categories_text)
            try:
                record = await client.call_structured(messages, EventRecord)
            except (json.JSONDecodeError, ValidationError):
                skipped_count += 1
                continue
            assert isinstance(record, EventRecord)

            slugs = _filter_categories(record.categories, valid_slugs)
            embedding = _embed_event_text(
                record.title, record.fact_summary, record.interpretation
            )

            event = Event(
                watch_id=source.watch_id,
                title=record.title,
                fact_summary=record.fact_summary,
                interpretation=record.interpretation,
                event_date=record.event_date,
                entities=record.entities,
                embedding=embedding,
            )
            session.add(event)
            # Flush so `event.id` (a Python-side `default=uuid.uuid4`,
            # populated only at flush time) is available for the
            # dependent `EventCategory`/`EventSource` rows below.
            await session.flush()

            session.add_all(
                EventCategory(event_id=event.id, category_id=category_id_by_slug[slug])
                for slug in slugs
            )
            session.add(EventSource(event_id=event.id, source_id=source.id))

            source.status = "processed"
            new_count += 1
            continue

        assert source.adjudication_decision == "existing"
        event_id = source.adjudication_event_id
        assert event_id is not None

        if source.adjudication_materiality == "material":
            event_result = await session.execute(select(Event).where(Event.id == event_id))
            event = event_result.scalar_one()

            messages = _build_messages(source, categories_text, existing_event=event)
            try:
                record = await client.call_structured(messages, EventRecord)
            except (json.JSONDecodeError, ValidationError):
                skipped_count += 1
                continue
            assert isinstance(record, EventRecord)

            slugs = _filter_categories(record.categories, valid_slugs)
            embedding = _embed_event_text(
                record.title, record.fact_summary, record.interpretation
            )

            event.title = record.title
            event.fact_summary = record.fact_summary
            event.interpretation = record.interpretation
            event.event_date = record.event_date
            event.entities = record.entities
            event.embedding = embedding
            event.last_material_update_at = datetime.now(UTC)

            await session.execute(
                delete(EventCategory).where(EventCategory.event_id == event.id)
            )
            session.add_all(
                EventCategory(event_id=event.id, category_id=category_id_by_slug[slug])
                for slug in slugs
            )
            session.add(EventSource(event_id=event.id, source_id=source.id))

            source.status = "processed"
            existing_updated_count += 1
        else:
            # `materiality in ("none", "minor")` -- no LLM call, no record
            # change; just link this source to the existing event.
            session.add(EventSource(event_id=event_id, source_id=source.id))
            source.status = "processed"
            existing_linked_count += 1

    return {
        "new": new_count,
        "existing_updated": existing_updated_count,
        "existing_linked": existing_linked_count,
        "skipped": skipped_count,
    }
