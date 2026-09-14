"""Tests for src/nie/pipeline/synthesize.py (issue #26).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_adjudicate.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test, not the module-level singleton, so pooled asyncpg connections stay
bound to this test's own event loop.

The LLM is stubbed exactly like `tests/test_pipeline_adjudicate.py`: a
real `LLMClient` is built with `Settings(_env_file=None,
llm_api_key="test-key", ...)` (no real network call, no real API key),
with `client._client.chat.completions.create` monkeypatched to an
`AsyncMock`, and that client is passed into
`synthesize_stage(session, client=...)` explicitly.

Embeddings are the **real** local `nie.embeddings.fastembed.embed_text`
output, not a stub -- per the issue's own test-list wording ("no need to
fake it, same precedent as any test exercising `nie.embeddings.fastembed`
directly"). No live network call results from this: the ONNX model is
loaded from fastembed's local cache, same narrow exception
`tests/test_embeddings_fastembed.py` already documents and relies on.

`Category` rows are seeded once per test via the real, idempotent
`nie.seed.categories.seed_categories` -- the 13 real slugs (including
`"other"`, needed for the category-fallback tests) rather than hand-rolled
fixture rows.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.config import Settings
from nie.db import create_engine, create_session_factory
from nie.embeddings.fastembed import embed_text
from nie.llm.client import LLMClient
from nie.models import Category, Event, EventCategory, EventSource, Source, Watch
from nie.pipeline.synthesize import synthesize_stage
from nie.seed.categories import seed_categories

REPO_ROOT = Path(__file__).parent.parent


def unique_slug(prefix: str) -> str:
    """A per-run-unique slug so tests stay independent of prior DB state."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def migrated_db() -> None:
    """Run `alembic upgrade head` against the live Compose DB.

    A plain (sync) fixture: Alembic drives its own event loop internally,
    so this must run outside pytest-asyncio's loop for the test function.
    Re-running this is idempotent -- Alembic is a no-op when the database
    is already at the target revision.
    """
    config = Config(str(REPO_ROOT / "alembic.ini"))
    upgrade(config, "head")


@pytest.fixture
async def session_factory(migrated_db: None) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory built from its own engine, scoped to this test."""
    engine = create_engine()
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
async def _seed_categories(session_factory: async_sessionmaker[AsyncSession]) -> None:
    """Ensure the 13 real seeded categories (including `"other"`) exist.

    `seed_categories` is its own idempotent upsert, so calling it again
    against a DB that already has these rows (from a prior test file's
    run) is a no-op.
    """
    async with session_factory() as session:
        await seed_categories(session)


@pytest.fixture(autouse=True)
async def _clear_stale_adjudicated_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Move every pre-existing `adjudicated` `source` row out of
    `synthesize_stage`'s selection.

    `synthesize_stage`'s selection query is global -- `status ==
    "adjudicated"`, no `watch_id` filter, by design -- and the Compose
    Postgres is shared and never truncated between test runs, so
    `tests/test_pipeline_adjudicate.py` deliberately leaves rows in
    exactly this state. Repointed to `status="processed"` rather than
    deleted, same precedent as that file's own stale-row fixture --
    `event_source`/other rows may reference these, and deleting would
    violate a foreign key.
    """
    async with session_factory() as session:
        await session.execute(
            update(Source).where(Source.status == "adjudicated").values(status="processed")
        )
        await session.commit()


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_base_url="https://example-llm.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeChatCompletion:
    """Duck-types the small slice of `openai`'s `ChatCompletion` we read."""

    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


def _client_with_stubbed_create() -> tuple[LLMClient, AsyncMock]:
    client = LLMClient(settings=_settings(), min_interval_seconds=0.0)
    stub_create = AsyncMock()
    client._client.chat.completions.create = stub_create  # type: ignore[method-assign]
    return client, stub_create


async def _make_watch(session: AsyncSession, prefix: str = "synthesize-test") -> uuid.UUID:
    watch = Watch(slug=unique_slug(prefix), name="Synthesize Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch.id


def _make_source(
    watch_id: uuid.UUID,
    *,
    status: str = "adjudicated",
    adjudication_decision: str | None = None,
    adjudication_event_id: uuid.UUID | None = None,
    adjudication_materiality: str | None = None,
    title: str = "Silver ETF inflows hit a record",
    content: str = "Body text about silver ETFs.",
) -> Source:
    return Source(
        watch_id=watch_id,
        url=f"https://example.com/{uuid.uuid4().hex[:8]}",
        title=title,
        source_name="Example",
        content=content,
        entities=[],
        status=status,
        embedding=[0.0] * 384,
        adjudication_decision=adjudication_decision,
        adjudication_event_id=adjudication_event_id,
        adjudication_materiality=adjudication_materiality,
    )


def _make_event(
    watch_id: uuid.UUID,
    *,
    title: str = "Some Existing Event",
    fact_summary: str = "Some fact summary.",
    interpretation: str = "Some interpretation.",
    event_date: datetime | None = None,
    entities: list[str] | None = None,
) -> Event:
    if event_date is None:
        event_date = datetime.now(UTC) - timedelta(days=1)
    return Event(
        watch_id=watch_id,
        title=title,
        fact_summary=fact_summary,
        interpretation=interpretation,
        event_date=event_date,
        entities=entities if entities is not None else ["Existing Entity"],
        embedding=embed_text(f"{title}\n\n{fact_summary}\n\n{interpretation}"),
    )


async def _fetch_source(
    session_factory: async_sessionmaker[AsyncSession], source_id: uuid.UUID
) -> Source:
    async with session_factory() as session:
        result = await session.execute(select(Source).where(Source.id == source_id))
        return result.scalar_one()


async def _fetch_event(
    session_factory: async_sessionmaker[AsyncSession], event_id: uuid.UUID
) -> Event:
    async with session_factory() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        return result.scalar_one()


async def _fetch_event_category_slugs(
    session_factory: async_sessionmaker[AsyncSession], event_id: uuid.UUID
) -> set[str]:
    async with session_factory() as session:
        result = await session.execute(
            select(Category.slug)
            .join(EventCategory, EventCategory.category_id == Category.id)
            .where(EventCategory.event_id == event_id)
        )
        return set(result.scalars().all())


async def _fetch_event_sources(
    session_factory: async_sessionmaker[AsyncSession], event_id: uuid.UUID
) -> list[uuid.UUID]:
    async with session_factory() as session:
        result = await session.execute(
            select(EventSource.source_id).where(EventSource.event_id == event_id)
        )
        return list(result.scalars().all())


async def test_synthesize_stage_creates_new_event(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"title": "Silver ETF Inflows Surge", '
        '"fact_summary": "ETF holdings rose sharply this week.", '
        '"interpretation": "Suggests renewed investor demand.", '
        '"event_date": "2026-01-05T00:00:00Z", '
        '"entities": ["iShares Silver Trust"], '
        '"categories": ["etf-investment", "demand"]}'
    )

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(
            watch_id, adjudication_decision="new", adjudication_materiality="material"
        )
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await synthesize_stage(session, client=client)
        await session.commit()

    assert result == {"new": 1, "existing_updated": 0, "existing_linked": 0, "skipped": 0}
    assert stub_create.call_count == 1

    row = await _fetch_source(session_factory, source_id)
    assert row.status == "processed"

    async with session_factory() as session:
        event_result = await session.execute(select(Event).where(Event.watch_id == watch_id))
        event = event_result.scalar_one()

    assert event.title == "Silver ETF Inflows Surge"
    assert event.fact_summary == "ETF holdings rose sharply this week."
    assert event.interpretation == "Suggests renewed investor demand."
    assert event.entities == ["iShares Silver Trust"]
    expected_embedding = embed_text(
        "Silver ETF Inflows Surge\n\n"
        "ETF holdings rose sharply this week.\n\n"
        "Suggests renewed investor demand."
    )
    assert list(event.embedding) == pytest.approx(expected_embedding)

    slugs = await _fetch_event_category_slugs(session_factory, event.id)
    assert slugs == {"etf-investment", "demand"}

    source_ids = await _fetch_event_sources(session_factory, event.id)
    assert source_ids == [source_id]


async def test_synthesize_stage_updates_existing_event_on_material_change(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch_id = await _make_watch(session)
        event = _make_event(
            watch_id,
            title="Old Title",
            fact_summary="Old fact summary.",
            interpretation="Old interpretation.",
            entities=["Old Entity"],
        )
        session.add(event)
        await session.commit()
        event_id = event.id
        # Seed a pre-existing category link that must be replaced, not
        # merely appended to.
        category_result = await session.execute(
            select(Category).where(Category.slug == "supply")
        )
        old_category = category_result.scalar_one()
        session.add(EventCategory(event_id=event_id, category_id=old_category.id))
        await session.commit()

    before = await _fetch_event(session_factory, event_id)
    before_title = before.title
    before_embedding = list(before.embedding)
    before_last_material_update_at = before.last_material_update_at

    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"title": "New Title", '
        '"fact_summary": "New fact summary.", '
        '"interpretation": "New interpretation.", '
        '"event_date": "2026-02-01T00:00:00Z", '
        '"entities": ["Old Entity", "New Entity"], '
        '"categories": ["mining"]}'
    )

    async with session_factory() as session:
        source = _make_source(
            watch_id,
            adjudication_decision="existing",
            adjudication_event_id=event_id,
            adjudication_materiality="material",
        )
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await synthesize_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing_updated": 1, "existing_linked": 0, "skipped": 0}
    assert stub_create.call_count == 1

    row = await _fetch_source(session_factory, source_id)
    assert row.status == "processed"

    after = await _fetch_event(session_factory, event_id)
    assert after.title == "New Title"
    assert after.title != before_title
    assert after.fact_summary == "New fact summary."
    assert after.interpretation == "New interpretation."
    assert after.entities == ["Old Entity", "New Entity"]
    assert list(after.embedding) != pytest.approx(before_embedding)
    assert after.last_material_update_at > before_last_material_update_at

    slugs = await _fetch_event_category_slugs(session_factory, event_id)
    assert slugs == {"mining"}

    source_ids = await _fetch_event_sources(session_factory, event_id)
    assert source_ids == [source_id]


@pytest.mark.parametrize("materiality", ["none", "minor"])
async def test_synthesize_stage_links_existing_event_without_llm_call(
    session_factory: async_sessionmaker[AsyncSession],
    materiality: str,
) -> None:
    async with session_factory() as session:
        watch_id = await _make_watch(session)
        event = _make_event(watch_id)
        session.add(event)
        await session.commit()
        event_id = event.id

    before = await _fetch_event(session_factory, event_id)
    before_title = before.title
    before_fact_summary = before.fact_summary
    before_interpretation = before.interpretation
    before_entities = list(before.entities)
    before_embedding = list(before.embedding)
    before_last_material_update_at = before.last_material_update_at
    before_slugs = await _fetch_event_category_slugs(session_factory, event_id)

    client, stub_create = _client_with_stubbed_create()

    async with session_factory() as session:
        source = _make_source(
            watch_id,
            adjudication_decision="existing",
            adjudication_event_id=event_id,
            adjudication_materiality=materiality,
        )
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await synthesize_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing_updated": 0, "existing_linked": 1, "skipped": 0}
    assert stub_create.call_count == 0

    row = await _fetch_source(session_factory, source_id)
    assert row.status == "processed"

    after = await _fetch_event(session_factory, event_id)
    assert after.title == before_title
    assert after.fact_summary == before_fact_summary
    assert after.interpretation == before_interpretation
    assert after.entities == before_entities
    assert list(after.embedding) == pytest.approx(before_embedding)
    assert after.last_material_update_at == before_last_material_update_at

    after_slugs = await _fetch_event_category_slugs(session_factory, event_id)
    assert after_slugs == before_slugs

    source_ids = await _fetch_event_sources(session_factory, event_id)
    assert source_ids == [source_id]


async def test_synthesize_stage_filters_invalid_category_slug_keeping_valid_ones(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"title": "Some Title", '
        '"fact_summary": "Some fact summary.", '
        '"interpretation": "Some interpretation.", '
        '"event_date": null, '
        '"entities": [], '
        '"categories": ["mining", "not-a-real-slug"]}'
    )

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(
            watch_id, adjudication_decision="new", adjudication_materiality="material"
        )
        session.add(source)
        await session.commit()

        result = await synthesize_stage(session, client=client)
        await session.commit()

    assert result == {"new": 1, "existing_updated": 0, "existing_linked": 0, "skipped": 0}

    async with session_factory() as session:
        event_result = await session.execute(select(Event).where(Event.watch_id == watch_id))
        event = event_result.scalar_one()

    slugs = await _fetch_event_category_slugs(session_factory, event.id)
    assert slugs == {"mining"}


async def test_synthesize_stage_falls_back_to_other_when_all_slugs_invalid(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"title": "Some Title", '
        '"fact_summary": "Some fact summary.", '
        '"interpretation": "Some interpretation.", '
        '"event_date": null, '
        '"entities": [], '
        '"categories": ["totally-fake-slug"]}'
    )

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(
            watch_id, adjudication_decision="new", adjudication_materiality="material"
        )
        session.add(source)
        await session.commit()

        result = await synthesize_stage(session, client=client)
        await session.commit()

    assert result == {"new": 1, "existing_updated": 0, "existing_linked": 0, "skipped": 0}

    async with session_factory() as session:
        event_result = await session.execute(select(Event).where(Event.watch_id == watch_id))
        event = event_result.scalar_one()

    slugs = await _fetch_event_category_slugs(session_factory, event.id)
    assert slugs == {"other"}


async def test_synthesize_stage_skips_row_malformed_on_both_attempts(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    # Structurally valid JSON but violates `EventRecord`'s `min_length=1`
    # constraint on `title` -> `ValidationError` on both `call_structured`
    # attempts.
    stub_create.side_effect = [
        _FakeChatCompletion(
            '{"title": "", "fact_summary": "x", "interpretation": "x", '
            '"event_date": null, "entities": [], "categories": ["other"]}'
        ),
        _FakeChatCompletion(
            '{"title": "", "fact_summary": "x", "interpretation": "x", '
            '"event_date": null, "entities": [], "categories": ["other"]}'
        ),
    ]

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(
            watch_id, adjudication_decision="new", adjudication_materiality="material"
        )
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await synthesize_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing_updated": 0, "existing_linked": 0, "skipped": 1}
    assert stub_create.call_count == 2

    row = await _fetch_source(session_factory, source_id)
    assert row.status == "adjudicated"

    async with session_factory() as session:
        event_result = await session.execute(select(Event).where(Event.watch_id == watch_id))
        assert event_result.scalars().all() == []


async def test_synthesize_stage_leaves_other_statuses_untouched(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        processed = _make_source(watch_id, status="processed")
        extracted = _make_source(watch_id, status="extracted")
        session.add_all([processed, extracted])
        await session.commit()
        processed_id = processed.id
        extracted_id = extracted.id

        result = await synthesize_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing_updated": 0, "existing_linked": 0, "skipped": 0}
    assert stub_create.call_count == 0

    processed_row = await _fetch_source(session_factory, processed_id)
    assert processed_row.status == "processed"

    extracted_row = await _fetch_source(session_factory, extracted_id)
    assert extracted_row.status == "extracted"


async def test_synthesize_stage_is_idempotent_across_two_calls(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"title": "Some Title", '
        '"fact_summary": "Some fact summary.", '
        '"interpretation": "Some interpretation.", '
        '"event_date": null, '
        '"entities": [], '
        '"categories": ["other"]}'
    )

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(
            watch_id, adjudication_decision="new", adjudication_materiality="material"
        )
        session.add(source)
        await session.commit()
        source_id = source.id

        first_result = await synthesize_stage(session, client=client)
        await session.commit()

    assert first_result == {"new": 1, "existing_updated": 0, "existing_linked": 0, "skipped": 0}
    assert stub_create.call_count == 1

    async with session_factory() as session:
        second_result = await synthesize_stage(session, client=client)
        await session.commit()

    assert second_result == {"new": 0, "existing_updated": 0, "existing_linked": 0, "skipped": 0}
    assert stub_create.call_count == 1

    row = await _fetch_source(session_factory, source_id)
    assert row.status == "processed"
