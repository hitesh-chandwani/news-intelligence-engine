"""Tests for src/nie/pipeline/triage.py (issue #24).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_extract.py`'s/
`tests/test_pipeline_embed.py`'s `migrated_db`/`session_factory` fixture
pattern -- a fresh engine per test, not the module-level singleton, so
pooled asyncpg connections stay bound to this test's own event loop.

Like `extract_stage`/`embed_stage`, `triage_stage` never calls
`Settings()` itself and never looks up the Silver watch by its fixed
slug -- it only queries `source` rows by `status`. So each test below
creates its own `Watch` row with `unique_slug` (same helper
`tests/test_pipeline_extract.py` defines) and attaches its `Source` rows
to that watch, keeping tests independent of each other and of any prior
run's leftover rows.

The LLM is stubbed exactly like `tests/test_llm_client.py`: a real
`LLMClient` is built with `Settings(_env_file=None, llm_api_key="test-key",
...)` (no real network call, no real API key), with
`client._client.chat.completions.create` monkeypatched to an `AsyncMock`,
and that client is passed into `triage_stage(session, client=...)`
explicitly -- there's no module-level function to monkeypatch the way
`extract_stage`'s tests patch `trafilatura`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
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
from nie.models import Source, Watch
from nie.pipeline.triage import triage_stage

REPO_ROOT = Path(__file__).parent.parent


def unique_slug(prefix: str) -> str:
    """A per-run-unique slug so tests stay independent of prior DB state.

    Same helper `tests/test_pipeline_extract.py` defines -- there's no
    row-cleanup fixture, so each test generates its own `Watch.slug`
    rather than relying on a fixed one.
    """
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
async def _clear_stale_extracted_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Move every pre-existing `extracted` `source` row out of
    `triage_stage`'s selection.

    `triage_stage`'s selection query (per the issue's stage contract) is
    global -- `status == "extracted"`, no `watch_id` filter, by design
    (every extracted source, regardless of which watch discovered it,
    needs a triage verdict). The Compose Postgres is shared and never
    truncated between test runs (see `tests/test_pipeline_discover.py`'s
    own module docstring), so `tests/test_pipeline_embed.py` and prior
    runs of this suite leave `extracted` rows sitting in the table
    indefinitely -- the same global-selection-query test-pollution
    `tests/test_pipeline_extract.py` (#20) and `tests/test_pipeline_embed.py`
    (#21) each hit for their own selection queries. Rows are repointed to
    `status="processed"` rather than deleted, same as
    `tests/test_pipeline_extract.py`'s fixture -- some historical rows are
    referenced by an `event_source` row, and deleting them would violate
    that foreign key.
    """
    async with session_factory() as session:
        await session.execute(
            update(Source).where(Source.status == "extracted").values(status="processed")
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


async def _make_watch(session: AsyncSession) -> uuid.UUID:
    watch = Watch(slug=unique_slug("triage-test"), name="Triage Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch.id


def _make_source(
    watch_id: uuid.UUID,
    url: str,
    *,
    title: str = "Some Title",
    content: str = "Some content.",
    status: str = "extracted",
) -> Source:
    return Source(
        watch_id=watch_id,
        url=url,
        title=title,
        source_name="Example",
        published_at=None,
        content=content,
        entities=[],
        status=status,
    )


async def test_triage_stage_keeps_source_on_positive_verdict(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"plausible": true, "note": "Plausibly relevant to Silver."}'
    )

    url = "https://example.com/silver-etf-inflows"

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(_make_source(watch_id, url, title="Silver ETF inflows hit a record"))
        await session.commit()

        result = await triage_stage(session, client=client)
        await session.commit()

    assert result == {"triaged_out": 0, "kept": 1, "skipped": 0}
    assert stub_create.call_count == 1

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extracted"
    assert row.triage_note is None


async def test_triage_stage_drops_source_on_negative_verdict(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"plausible": false, "note": "About a silver anniversary, not the metal."}'
    )

    url = "https://example.com/silver-anniversary"

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(
            _make_source(watch_id, url, title="Couple celebrates silver wedding anniversary")
        )
        await session.commit()

        result = await triage_stage(session, client=client)
        await session.commit()

    assert result == {"triaged_out": 1, "kept": 0, "skipped": 0}

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "triaged_out"
    assert row.triage_note == "About a silver anniversary, not the metal."


async def test_triage_stage_handles_mixed_batch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    keep_url = "https://example.com/silver-mine-supply-cut"
    drop_url = "https://example.com/unrelated-noise"

    def fake_create(*, messages: list[dict[str, str]], **_: object) -> _FakeChatCompletion:
        # `triage.md`'s fixed framing text itself mentions "mine supply", so
        # discriminate on a phrase that only ever appears in one row's own
        # title/content, never in the shared boilerplate.
        prompt = messages[0]["content"]
        if "weekend" in prompt.lower():
            return _FakeChatCompletion('{"plausible": false, "note": "Not about Silver."}')
        return _FakeChatCompletion('{"plausible": true, "note": "Mine supply news."}')

    stub_create.side_effect = fake_create

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add_all(
            [
                _make_source(watch_id, keep_url, title="Major Silver mine supply cut announced"),
                _make_source(watch_id, drop_url, title="Local weather forecast for the weekend"),
            ]
        )
        await session.commit()

        result = await triage_stage(session, client=client)
        await session.commit()

    assert result == {"triaged_out": 1, "kept": 1, "skipped": 0}
    assert stub_create.call_count == 2

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        by_url = {row.url: row for row in rows_result.scalars().all()}

    assert by_url[keep_url].status == "extracted"
    assert by_url[keep_url].triage_note is None
    assert by_url[drop_url].status == "triaged_out"
    assert by_url[drop_url].triage_note == "Not about Silver."


async def test_triage_stage_skips_row_that_fails_validation_twice(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    # Missing the required "note" field on both attempts -> ValidationError
    # on `call_structured`'s first try and again on its one retry.
    stub_create.side_effect = [
        _FakeChatCompletion('{"plausible": true}'),
        _FakeChatCompletion('{"plausible": true}'),
    ]

    url = "https://example.com/malformed-response"

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(_make_source(watch_id, url))
        await session.commit()

        result = await triage_stage(session, client=client)
        await session.commit()

    assert result == {"triaged_out": 0, "kept": 0, "skipped": 1}
    assert stub_create.call_count == 2

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extracted"
    assert row.triage_note is None
