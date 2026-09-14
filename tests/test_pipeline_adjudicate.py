"""Tests for src/nie/pipeline/adjudicate.py (issue #25).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_triage.py`'s/
`tests/test_pipeline_match.py`'s `migrated_db`/`session_factory` fixture
pattern -- a fresh engine per test, not the module-level singleton, so
pooled asyncpg connections stay bound to this test's own event loop.

The LLM is stubbed exactly like `tests/test_pipeline_triage.py`: a real
`LLMClient` is built with `Settings(_env_file=None, llm_api_key="test-key",
...)` (no real network call, no real API key), with
`client._client.chat.completions.create` monkeypatched to an `AsyncMock`,
and that client is passed into `adjudicate_stage(session, client=...)`
explicitly.

`find_candidate_events` (#22) builds its own `Settings()` internally, so
`_isolate_env` (copied from `tests/test_pipeline_match.py`) keeps the
real shell/`.env` from leaking `DEDUP_WINDOW_DAYS` into these tests -- the
default `dedup_window_days=14` is what every fixture below is built
against. Every embedding used is a hand-authored fixed 384-length float
list, same precedent as `tests/test_pipeline_match.py` -- no live network
call, no real embedding model.
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
from nie.llm.client import LLMClient
from nie.models import Event, Source, Watch
from nie.pipeline.adjudicate import adjudicate_stage

REPO_ROOT = Path(__file__).parent.parent

EMBEDDING_DIM = 384

# Same list `tests/test_pipeline_match.py` uses, so the real shell/`.env`
# never leaks into `Settings()` calls made inside `find_candidate_events`.
_ALL_ENV_VARS = [
    "DATABASE_URL",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "RESEND_API_KEY",
    "NOTIFY_EMAIL_TO",
    "DISCOVERY_PROVIDERS",
    "RSS_FEEDS",
    "POLL_INTERVAL_MINUTES",
    "NOTIFY_MIN_IMPORTANCE",
    "DEDUP_WINDOW_DAYS",
]


def unique_slug(prefix: str) -> str:
    """A per-run-unique slug so tests stay independent of prior DB state."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _vector(**components: float) -> list[float]:
    """A fixed 384-length float list with the given indices set and every
    other component zero -- a hand-authored stand-in for a real
    embedding, never a live FastEmbed call.
    """
    vec = [0.0] * EMBEDDING_DIM
    for index, value in components.items():
        vec[int(index)] = value
    return vec


# The source's embedding: a unit vector along axis 0.
SOURCE_EMBEDDING = _vector(**{"0": 1.0})
# Identical direction -- cosine distance 0, so this event is always the
# nearest candidate for `SOURCE_EMBEDDING`.
NEAR_EMBEDDING = _vector(**{"0": 1.0})


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the real shell environment never leaks into these tests."""
    for name in _ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


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
async def _clear_stale_extracted_and_embedded_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Move every pre-existing `extracted` + embedded `source` row out of
    `adjudicate_stage`'s selection.

    `adjudicate_stage`'s selection query (per the issue's stage contract)
    is global -- `status == "extracted" AND embedding IS NOT NULL`, no
    `watch_id` filter, by design. The Compose Postgres is shared and never
    truncated between test runs, so `tests/test_pipeline_embed.py` (#21)
    deliberately leaves rows in exactly this state (its own docstring:
    "an embedded row just gains an embedding and stays
    `status='extracted'`") -- the same global-selection-query test-
    pollution `tests/test_pipeline_triage.py`/`test_pipeline_embed.py`
    each hit for their own selection queries. Rows are repointed to
    `status="processed"` rather than deleted, same precedent as those
    files' fixtures -- some historical rows may be referenced by an
    `event_source` row, and deleting them would violate that foreign key.
    """
    async with session_factory() as session:
        await session.execute(
            update(Source)
            .where(Source.status == "extracted", Source.embedding.isnot(None))
            .values(status="processed")
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


async def _make_watch(session: AsyncSession, prefix: str = "adjudicate-test") -> uuid.UUID:
    watch = Watch(slug=unique_slug(prefix), name="Adjudicate Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch.id


def _make_source(
    watch_id: uuid.UUID,
    *,
    embedding: list[float] | None = SOURCE_EMBEDDING,
    status: str = "extracted",
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
        embedding=embedding,
    )


def _make_event(
    watch_id: uuid.UUID,
    *,
    embedding: list[float] = NEAR_EMBEDDING,
    event_date: datetime | None = None,
    title: str = "Some Existing Event",
) -> Event:
    if event_date is None:
        event_date = datetime.now(UTC) - timedelta(days=1)
    return Event(
        watch_id=watch_id,
        title=title,
        fact_summary="Some fact summary.",
        interpretation="Some interpretation.",
        event_date=event_date,
        entities=[],
        embedding=embedding,
    )


async def _fetch(session_factory: async_sessionmaker[AsyncSession], source_id: uuid.UUID) -> Source:
    async with session_factory() as session:
        result = await session.execute(select(Source).where(Source.id == source_id))
        return result.scalar_one()


async def test_adjudicate_stage_persists_new_decision(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"decision": "new", "event_id": null, "materiality": "material"}'
    )

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(watch_id)
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert result == {"new": 1, "existing": 0, "noise": 0, "skipped": 0}

    row = await _fetch(session_factory, source_id)
    assert row.status == "adjudicated"
    assert row.adjudication_decision == "new"
    assert row.adjudication_materiality == "material"
    assert row.adjudication_event_id is None


async def test_adjudicate_stage_persists_noise_decision(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"decision": "noise", "event_id": null, "materiality": "none"}'
    )

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(watch_id)
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing": 0, "noise": 1, "skipped": 0}

    row = await _fetch(session_factory, source_id)
    assert row.status == "processed"
    assert row.adjudication_decision == "noise"
    assert row.adjudication_materiality == "none"
    assert row.adjudication_event_id is None


@pytest.mark.parametrize("materiality", ["none", "minor", "material"])
async def test_adjudicate_stage_persists_existing_decision_for_each_materiality(
    session_factory: async_sessionmaker[AsyncSession],
    materiality: str,
) -> None:
    async with session_factory() as session:
        watch_id = await _make_watch(session)
        event = _make_event(watch_id)
        session.add(event)
        await session.commit()
        event_id = event.id

    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        f'{{"decision": "existing", "event_id": "{event_id}", "materiality": "{materiality}"}}'
    )

    async with session_factory() as session:
        source = _make_source(watch_id)
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing": 1, "noise": 0, "skipped": 0}

    row = await _fetch(session_factory, source_id)
    assert row.status == "adjudicated"
    assert row.adjudication_decision == "existing"
    assert row.adjudication_event_id == event_id
    assert row.adjudication_materiality == materiality


async def test_adjudicate_stage_skips_row_malformed_on_both_attempts(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    # Structurally valid JSON but violates the decision/materiality
    # pairing rule (`new` requires `materiality == "material"`) on both
    # `call_structured` attempts -> `ValidationError` both times.
    stub_create.side_effect = [
        _FakeChatCompletion('{"decision": "new", "event_id": null, "materiality": "minor"}'),
        _FakeChatCompletion('{"decision": "new", "event_id": null, "materiality": "minor"}'),
    ]

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(watch_id)
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing": 0, "noise": 0, "skipped": 1}
    assert stub_create.call_count == 2

    row = await _fetch(session_factory, source_id)
    assert row.status == "extracted"
    assert row.adjudication_decision is None
    assert row.adjudication_materiality is None
    assert row.adjudication_event_id is None


async def test_adjudicate_stage_skips_row_with_hallucinated_event_id(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch_id = await _make_watch(session)
        event = _make_event(watch_id)
        session.add(event)
        await session.commit()

    hallucinated_event_id = uuid.uuid4()
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        f'{{"decision": "existing", "event_id": "{hallucinated_event_id}", '
        '"materiality": "minor"}'
    )

    async with session_factory() as session:
        source = _make_source(watch_id)
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing": 0, "noise": 0, "skipped": 1}

    row = await _fetch(session_factory, source_id)
    assert row.status == "extracted"
    assert row.adjudication_decision is None


async def test_adjudicate_stage_skips_existing_decision_with_no_candidates(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # No `Event` rows at all for this watch -> `find_candidate_events`
    # returns `[]`, so no `event_id` can be a member of the candidate set.
    made_up_event_id = uuid.uuid4()
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        f'{{"decision": "existing", "event_id": "{made_up_event_id}", "materiality": "none"}}'
    )

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(watch_id)
        session.add(source)
        await session.commit()
        source_id = source.id

        result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing": 0, "noise": 0, "skipped": 1}

    row = await _fetch(session_factory, source_id)
    assert row.status == "extracted"
    assert row.adjudication_decision is None


async def test_adjudicate_stage_leaves_other_statuses_untouched(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        triaged_out = _make_source(watch_id, status="triaged_out")
        no_embedding = _make_source(watch_id, embedding=None, status="extracted")
        session.add_all([triaged_out, no_embedding])
        await session.commit()
        triaged_out_id = triaged_out.id
        no_embedding_id = no_embedding.id

        result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert result == {"new": 0, "existing": 0, "noise": 0, "skipped": 0}
    assert stub_create.call_count == 0

    triaged_out_row = await _fetch(session_factory, triaged_out_id)
    assert triaged_out_row.status == "triaged_out"
    assert triaged_out_row.adjudication_decision is None

    no_embedding_row = await _fetch(session_factory, no_embedding_id)
    assert no_embedding_row.status == "extracted"
    assert no_embedding_row.embedding is None
    assert no_embedding_row.adjudication_decision is None


async def test_adjudicate_stage_is_idempotent_across_two_calls(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"decision": "new", "event_id": null, "materiality": "material"}'
    )

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        source = _make_source(watch_id)
        session.add(source)
        await session.commit()
        source_id = source.id

        first_result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert first_result == {"new": 1, "existing": 0, "noise": 0, "skipped": 0}
    assert stub_create.call_count == 1

    async with session_factory() as session:
        second_result = await adjudicate_stage(session, client=client)
        await session.commit()

    assert second_result == {"new": 0, "existing": 0, "noise": 0, "skipped": 0}
    # No new LLM call was made -- the row is no longer selected at all.
    assert stub_create.call_count == 1

    row = await _fetch(session_factory, source_id)
    assert row.status == "adjudicated"
    assert row.adjudication_decision == "new"
