"""Tests for src/nie/pipeline/embed.py (issue #21).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_extract.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test, not the module-level singleton, so pooled asyncpg connections stay
bound to this test's own event loop.

Like `extract_stage`, `embed_stage` never calls `Settings()` and never
looks up the Silver watch by its fixed slug -- it only queries `source`
rows by `status`/`embedding`. So each test below creates its own `Watch`
row with `unique_slug` (same helper `tests/test_pipeline_extract.py`
defines) and attaches its `Source` rows to that watch, keeping tests
independent of each other and of any prior run's leftover rows (the
Compose Postgres is never truncated between test runs, per
`tests/test_pipeline_discover.py`'s own module docstring).

`nie.pipeline.embed.embed_texts` is monkeypatched to a deterministic stub
-- per issue #21's "Testing the real model" section, this file keeps
following the no-live-network rule exactly (no real model load); proving
fastembed's real output is `tests/test_embeddings_fastembed.py`'s job,
not this one's. The stub records every batch of texts it's called with
(so a test can assert exactly what was/wasn't passed to it) and returns
one distinguishable 384-length vector per input text, in order.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Source, Watch
from nie.pipeline import embed as embed_module
from nie.pipeline.embed import embed_stage

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
async def _clear_stale_extracted_rows_without_embedding(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Give every pre-existing `extracted` row with no `embedding` a dummy
    embedding so it drops out of `embed_stage`'s selection.

    `embed_stage`'s selection query (per the issue's stage contract) is
    global -- `status == "extracted" AND embedding IS NULL`, no
    `watch_id` filter, by design (the stage must catch up on any
    extracted-but-unembedded row regardless of which watch discovered
    it). The Compose Postgres is shared and never truncated between test
    runs (see `tests/test_pipeline_discover.py`'s own module docstring),
    so `tests/test_pipeline_extract.py` and prior runs of this suite
    leave `extracted` rows with `embedding IS NULL` sitting in the table
    indefinitely -- the same global-selection-query test-pollution
    `tests/test_pipeline_extract.py` hit for `discovered`/
    `extract_failed` rows (#20). Unlike that fixture, this one can't
    repoint `status` -- an `extracted` row with no `embedding` yet is a
    perfectly valid, common state outside tests -- so it gives these rows
    a dummy, non-null embedding instead, which is exactly what removes a
    row from `embed_stage`'s selection without touching its status.
    """
    async with session_factory() as session:
        await session.execute(
            update(Source)
            .where(Source.status == "extracted", Source.embedding.is_(None))
            .values(embedding=[0.0] * 384)
        )
        await session.commit()


def _stub_embed_texts(
    calls: list[list[str]],
) -> Callable[[list[str]], list[list[float]]]:
    """Build an `embed_texts` stub that records every batch it's called
    with and returns one distinguishable 384-length vector per input
    text, in order -- so a test can assert both "was this text embedded"
    and "did the Nth result land on the Nth row".
    """

    def embed_texts(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(index)] * 384 for index in range(len(texts))]

    return embed_texts


async def _make_watch(session: AsyncSession) -> uuid.UUID:
    watch = Watch(slug=unique_slug("embed-test"), name="Embed Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch.id


def _make_source(
    watch_id: uuid.UUID,
    url: str,
    *,
    status: str,
    title: str = "Some Title",
    content: str | None = "Some content.",
    embedding: list[float] | None = None,
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
        embedding=embedding,
    )


async def test_embed_stage_embeds_extracted_row_with_no_embedding(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(embed_module, "embed_texts", _stub_embed_texts(calls))

    url = "https://example.com/needs-embedding"

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(
            _make_source(
                watch_id,
                url,
                status="extracted",
                title="Silver ETF inflows hit a record",
                content="Body text about silver ETFs.",
            )
        )
        await session.commit()

        result = await embed_stage(session)
        await session.commit()

    assert result == {"embedded": 1}
    assert calls == [["Silver ETF inflows hit a record\n\nBody text about silver ETFs."]]

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extracted"
    assert row.embedding is not None
    assert len(row.embedding) == 384


async def test_embed_stage_leaves_already_embedded_row_untouched(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(embed_module, "embed_texts", _stub_embed_texts(calls))

    url = "https://example.com/already-embedded"
    existing_embedding = [0.5] * 384

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(
            _make_source(
                watch_id,
                url,
                status="extracted",
                embedding=existing_embedding,
            )
        )
        await session.commit()

        result = await embed_stage(session)
        await session.commit()

    assert result == {"embedded": 0}
    assert calls == []

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.status == "extracted"
    assert row.embedding is not None
    assert list(row.embedding) == pytest.approx(existing_embedding)


async def test_embed_stage_leaves_other_statuses_untouched(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(embed_module, "embed_texts", _stub_embed_texts(calls))

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add_all(
            [
                _make_source(
                    watch_id, "https://example.com/discovered", status="discovered", content=None
                ),
                _make_source(
                    watch_id,
                    "https://example.com/extract-failed",
                    status="extract_failed",
                    content=None,
                ),
                _make_source(
                    watch_id, "https://example.com/triaged-out", status="triaged_out"
                ),
                _make_source(watch_id, "https://example.com/processed", status="processed"),
            ]
        )
        await session.commit()

        result = await embed_stage(session)
        await session.commit()

    assert result == {"embedded": 0}
    assert calls == []

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        by_url = {row.url: row for row in rows_result.scalars().all()}

    assert by_url["https://example.com/discovered"].status == "discovered"
    assert by_url["https://example.com/discovered"].embedding is None
    assert by_url["https://example.com/extract-failed"].status == "extract_failed"
    assert by_url["https://example.com/extract-failed"].embedding is None
    assert by_url["https://example.com/triaged-out"].status == "triaged_out"
    assert by_url["https://example.com/triaged-out"].embedding is None
    assert by_url["https://example.com/processed"].status == "processed"
    assert by_url["https://example.com/processed"].embedding is None


async def test_embed_stage_assigns_vectors_back_in_order(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(embed_module, "embed_texts", _stub_embed_texts(calls))

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add_all(
            [
                _make_source(
                    watch_id, "https://example.com/first", status="extracted", title="First"
                ),
                _make_source(
                    watch_id, "https://example.com/second", status="extracted", title="Second"
                ),
            ]
        )
        await session.commit()

        result = await embed_stage(session)
        await session.commit()

    assert result == {"embedded": 2}
    assert len(calls) == 1
    assert calls[0] == [
        "First\n\nSome content.",
        "Second\n\nSome content.",
    ]

    async with session_factory() as session:
        rows_result = await session.execute(
            select(Source).where(Source.watch_id == watch_id).order_by(Source.title)
        )
        rows = rows_result.scalars().all()

    first_row = next(row for row in rows if row.title == "First")
    second_row = next(row for row in rows if row.title == "Second")
    assert first_row.embedding is not None
    assert second_row.embedding is not None
    assert list(first_row.embedding) == pytest.approx([0.0] * 384)
    assert list(second_row.embedding) == pytest.approx([1.0] * 384)


async def test_embed_stage_is_idempotent_across_two_calls(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(embed_module, "embed_texts", _stub_embed_texts(calls))

    url = "https://example.com/idempotent"

    async with session_factory() as session:
        watch_id = await _make_watch(session)
        session.add(_make_source(watch_id, url, status="extracted"))
        await session.commit()

        first_result = await embed_stage(session)
        await session.commit()

    assert first_result == {"embedded": 1}
    assert len(calls) == 1

    async with session_factory() as session:
        second_result = await embed_stage(session)
        await session.commit()

    assert second_result == {"embedded": 0}
    assert len(calls) == 1

    async with session_factory() as session:
        rows_result = await session.execute(select(Source).where(Source.watch_id == watch_id))
        row = rows_result.scalar_one()

    assert row.embedding is not None
    assert len(row.embedding) == 384
