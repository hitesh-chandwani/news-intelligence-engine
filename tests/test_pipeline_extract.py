"""Tests for src/nie/pipeline/extract.py (issue #20).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_discover.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test, not the module-level singleton, so pooled asyncpg connections stay
bound to this test's own event loop.

Unlike `discover_stage`, `extract_stage` never looks up the Silver watch
by its fixed slug -- it only queries `source` rows by `status`. So each
test below creates its own `Watch` row with `unique_slug` (same helper
`tests/test_models.py` defines) and attaches its `Source` rows to that
watch, keeping tests independent of each other and of any prior run's
leftover rows (the Compose Postgres is never truncated between test
runs, per `tests/test_pipeline_discover.py`'s own module docstring).

As of #45, `extract_stage` *does* call `Settings()` internally (for
`max_extract_attempts`), same "construct `Settings()` inside the stage"
pattern `discover_stage` uses -- so, same as
`tests/test_pipeline_discover.py`'s `_isolate_env` fixture, `MAX_EXTRACT_
ATTEMPTS` is deleted from the environment on every test here to keep the
real shell/`.env` from leaking in and changing the cap tests rely on.

`trafilatura.fetch_url` is monkeypatched through `nie.sources.
extract_trafilatura.trafilatura`, same pattern as
`tests/test_extract_trafilatura.py` -- no test here makes a live network
call. A `fetch_url` stub is shared across tests: it records every URL
it's called with (so a test can assert the extractor was/wasn't called
for a given URL) and returns real fixture HTML for a "success" URL,
`None` for a "failure" URL, and raises for anything else so an
unexpected call fails loudly instead of silently returning junk.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.config import Settings
from nie.db import create_engine, create_session_factory
from nie.models import Source, Watch
from nie.pipeline.extract import extract_stage
from nie.sources import extract_trafilatura

REPO_ROOT = Path(__file__).parent.parent
FIXTURE_HTML = (
    REPO_ROOT / "tests" / "fixtures" / "sources" / "silver-etf-inflows-article.html"
).read_text()

SUCCESS_URL = "https://example.com/silver-etf-inflows"
FAILURE_URL = "https://example.com/unparseable-article"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real shell/`.env` from leaking into `extract_stage`'s
    internal `Settings()` call (#45), same `_isolate_env` pattern
    `tests/test_pipeline_discover.py` uses for `discover_stage`.

    Only `MAX_EXTRACT_ATTEMPTS` is deleted -- it's the only `Settings`
    field `extract_stage` reads -- so a stray `MAX_EXTRACT_ATTEMPTS` in
    the environment can't desync these tests from the cap value the
    `max_extract_attempts` fixture below reads back off `Settings()`.
    """
    monkeypatch.delenv("MAX_EXTRACT_ATTEMPTS", raising=False)


@pytest.fixture
def max_extract_attempts(_isolate_env: None) -> int:
    """The effective `Settings().max_extract_attempts` cap for these tests.

    Read off a real `Settings()` instance (with the environment already
    isolated by `_isolate_env`) rather than hardcoded, so these tests stay
    correct against whatever `design.md`-listed default `Settings` defines
    -- currently `3`.
    """
    return Settings().max_extract_attempts


def unique_slug(prefix: str) -> str:
    """A per-run-unique slug so tests stay independent of prior DB state.

    Same helper `tests/test_models.py` defines -- there's no row-cleanup
    fixture, so each test generates its own `Watch.slug` rather than
    relying on a fixed one.
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
async def _clear_stale_discovered_and_extract_failed_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Move every pre-existing `discovered`/`extract_failed` `source` row
    out of `extract_stage`'s selection.

    `extract_stage`'s selection query (per the issue's stage contract) is
    global -- `status IN ("discovered", "extract_failed")`, no `watch_id`
    filter, by design (a future provider-supplied-content row must reach
    this stage regardless of which watch discovered it). The Compose
    Postgres is shared and never truncated between test runs (see
    `tests/test_pipeline_discover.py`'s module docstring), so other test
    files -- and prior runs of this one -- can leave rows in these two
    statuses sitting in the table indefinitely, which would otherwise make
    this file's exact-count assertions non-deterministic.

    Rows are updated to `status="processed"` rather than deleted: some
    historical rows are referenced by an `event_source` row (from
    `tests/test_models.py`'s `Event`/`EventSource` tests), and deleting
    them would violate that foreign key. Repointing `status` doesn't touch
    any primary key, so it's always safe.
    """
    async with session_factory() as session:
        await session.execute(
            update(Source)
            .where(Source.status.in_(("discovered", "extract_failed")))
            .values(status="processed")
        )
        await session.commit()


def _fake_fetch_url(calls: list[str]) -> Callable[[str], str | None]:
    """Build a `trafilatura.fetch_url` stub that records every call.

    Resolves `SUCCESS_URL` to real fixture HTML, `FAILURE_URL` to `None`
    (fetch failure -> `ExtractionError`), and raises `AssertionError` for
    any other URL, so a bug that calls the extractor for the wrong row
    (e.g. a provider-supplied-content row that should have been skipped)
    fails the test loudly rather than silently.
    """

    def fetch_url(url: str) -> str | None:
        calls.append(url)
        if url == SUCCESS_URL:
            return FIXTURE_HTML
        if url == FAILURE_URL:
            return None
        raise AssertionError(f"unexpected fetch_url call for {url!r}")

    return fetch_url


async def _make_watch(session: AsyncSession) -> uuid.UUID:
    watch = Watch(slug=unique_slug("extract-test"), name="Extract Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch.id


def _make_source(
    watch_id: uuid.UUID,
    url: str,
    *,
    status: str,
    content: str | None = None,
    extracted_at: datetime | None = None,
    extract_attempts: int = 0,
) -> Source:
    return Source(
        watch_id=watch_id,
        url=url,
        title="Some Title",
        source_name="Example",
        published_at=None,
        content=content,
        entities=[],
        status=status,
        extracted_at=extracted_at,
        extract_attempts=extract_attempts,
    )


async def test_extract_stage_handles_success_and_failure_mix(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", _fake_fetch_url(calls))

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add_all(
            [
                _make_source(watch_id, SUCCESS_URL, status="discovered"),
                _make_source(watch_id, FAILURE_URL, status="discovered"),
            ]
        )
        await session.commit()

        result = await extract_stage(session)
        await session.commit()

    assert result == {"extracted": 1, "extract_failed": 1, "extract_capped": 0}

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        by_url = {row.url: row for row in rows_result.scalars().all()}

    success_row = by_url[SUCCESS_URL]
    assert success_row.status == "extracted"
    assert success_row.content is not None
    assert "largest weekly inflow on record" in success_row.content
    assert success_row.extracted_at is not None

    failure_row = by_url[FAILURE_URL]
    assert failure_row.status == "extract_failed"
    assert failure_row.content is None
    assert failure_row.extracted_at is None


async def test_extract_stage_skips_provider_supplied_content(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", _fake_fetch_url(calls))

    provider_url = "https://example.com/provider-supplied-content"

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(
            _make_source(
                watch_id,
                provider_url,
                status="discovered",
                content="Already have the full text, thanks.",
            )
        )
        await session.commit()

        result = await extract_stage(session)
        await session.commit()

    assert result == {"extracted": 1, "extract_failed": 0, "extract_capped": 0}
    assert provider_url not in calls

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extracted"
    assert row.content == "Already have the full text, thanks."
    assert row.extracted_at is not None


async def test_extract_stage_retries_extract_failed_row_on_next_call(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", _fake_fetch_url(calls))

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(_make_source(watch_id, FAILURE_URL, status="discovered"))
        await session.commit()

        first_result = await extract_stage(session)
        await session.commit()

    assert first_result == {"extracted": 0, "extract_failed": 1, "extract_capped": 0}
    assert calls.count(FAILURE_URL) == 1

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row_after_first = rows_result.scalar_one()

    # A source failing once has extract_attempts=1 after the call (well
    # below max_extract_attempts=3), so it's still eligible for retry.
    assert row_after_first.extract_attempts == 1
    assert row_after_first.status == "extract_failed"

    async with session_factory() as session:
        second_result = await extract_stage(session)
        await session.commit()

    # Same source row (still status="extract_failed") is retried: the mock
    # call count for its URL increases, and it's counted again this call.
    assert second_result == {"extracted": 0, "extract_failed": 1, "extract_capped": 0}
    assert calls.count(FAILURE_URL) == 2

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extract_failed"
    assert row.content is None
    assert row.extracted_at is None
    assert row.extract_attempts == 2


async def test_extract_stage_leaves_other_statuses_untouched(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", _fake_fetch_url(calls))

    extracted_at = datetime(2026, 1, 1, tzinfo=UTC)

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add_all(
            [
                _make_source(
                    watch_id,
                    "https://example.com/already-extracted",
                    status="extracted",
                    content="Existing content",
                    extracted_at=extracted_at,
                ),
                _make_source(
                    watch_id,
                    "https://example.com/triaged-out",
                    status="triaged_out",
                ),
                _make_source(
                    watch_id,
                    "https://example.com/processed",
                    status="processed",
                ),
            ]
        )
        await session.commit()

        result = await extract_stage(session)
        await session.commit()

    assert result == {"extracted": 0, "extract_failed": 0, "extract_capped": 0}
    assert calls == []

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        by_url = {row.url: row for row in rows_result.scalars().all()}

    already_extracted = by_url["https://example.com/already-extracted"]
    assert already_extracted.status == "extracted"
    assert already_extracted.content == "Existing content"
    assert already_extracted.extracted_at == extracted_at

    assert by_url["https://example.com/triaged-out"].status == "triaged_out"
    assert by_url["https://example.com/processed"].status == "processed"


async def test_extract_stage_skips_source_already_at_cap(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    max_extract_attempts: int,
) -> None:
    """A `status="extract_failed"` row already at the cap (#45) is excluded
    by the selection query entirely: the extractor is never called for its
    URL, its `extract_attempts` is left unchanged, and it's counted under
    `extract_capped`, not `extract_failed`.
    """
    calls: list[str] = []
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", _fake_fetch_url(calls))

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(
            _make_source(
                watch_id,
                FAILURE_URL,
                status="extract_failed",
                extract_attempts=max_extract_attempts,
            )
        )
        await session.commit()

        result = await extract_stage(session)
        await session.commit()

    assert result == {"extracted": 0, "extract_failed": 0, "extract_capped": 1}
    assert FAILURE_URL not in calls

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extract_failed"
    assert row.extract_attempts == max_extract_attempts


async def test_extract_stage_excludes_source_after_reaching_cap_boundary(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    max_extract_attempts: int,
) -> None:
    """Boundary case (#45): a source that fails exactly
    `max_extract_attempts` times across that many separate `extract_stage`
    calls ends the sequence with `extract_attempts == max_extract_attempts`
    and `status == "extract_failed"`, still having been retried on every
    one of those calls -- then is excluded starting on the *next* call,
    the transition from "still retried" to "capped," not just the two
    steady-state cases the other cap tests cover.
    """
    calls: list[str] = []
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", _fake_fetch_url(calls))

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(_make_source(watch_id, FAILURE_URL, status="discovered"))
        await session.commit()

    for attempt in range(1, max_extract_attempts + 1):
        async with session_factory() as session:
            result = await extract_stage(session)
            await session.commit()
        assert result == {"extracted": 0, "extract_failed": 1, "extract_capped": 0}
        assert calls.count(FAILURE_URL) == attempt

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extract_failed"
    assert row.extract_attempts == max_extract_attempts

    # One call past the cap: the row is now excluded rather than retried --
    # no new call to the extractor for its URL, and it's counted under
    # extract_capped this time.
    async with session_factory() as session:
        next_result = await extract_stage(session)
        await session.commit()

    assert next_result == {"extracted": 0, "extract_failed": 0, "extract_capped": 1}
    assert calls.count(FAILURE_URL) == max_extract_attempts

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extract_failed"
    assert row.extract_attempts == max_extract_attempts
