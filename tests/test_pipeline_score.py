"""Tests for src/nie/pipeline/score.py and match.py's find_nearest_events
(issue #27).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_match.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test, not the module-level singleton, so pooled asyncpg connections stay
bound to this test's own event loop.

Neither `find_nearest_events` nor `build_context_bundle` makes a live
network or LLM call. Every embedding used below is a hand-authored fixed
384-length float list (mostly zeros with one or two components set), same
precedent as `tests/test_pipeline_match.py` -- these tests exercise the
SQL/filtering/merge/dedup logic, not embedding quality. The feedback test
seeds the real category taxonomy via `nie.seed.categories.seed_categories`
(idempotent) and reuses two of its existing slugs, rather than inserting
new `Category` rows -- `tests/test_models.py`/`tests/test_run.py` assert
the table holds exactly the 13 seeded categories, and the Compose Postgres
is never truncated between test runs, so a hand-rolled extra `Category`
row here would permanently pollute that global count for every other
test file.

Each test creates its own `Watch` row(s) with `unique_slug` (same helper
`tests/test_models.py`/`tests/test_pipeline_match.py` define), keeping
tests independent of each other and of any prior run's leftover rows.
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
from nie.models import Category, ContextItem, Event, EventCategory, EventRelation, Feedback, Watch
from nie.pipeline.match import MatchCandidate, find_nearest_events
from nie.pipeline.score import (
    ContextBundle,
    FeedbackBucket,
    RelatedEvent,
    build_context_bundle,
    score_stage,
)
from nie.seed.categories import seed_categories

REPO_ROOT = Path(__file__).parent.parent

EMBEDDING_DIM = 384


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


# Same fixed vectors and distances `tests/test_pipeline_match.py` uses.
ANCHOR_EMBEDDING = _vector(**{"0": 1.0})
DIST_0_EMBEDDING = _vector(**{"0": 1.0})  # cosine distance 0
DIST_1_EMBEDDING = _vector(**{"0": 1.0, "1": 1.0})  # cosine distance 1 - 1/sqrt(2) ~= 0.29289
DIST_2_EMBEDDING = _vector(**{"0": 1.0, "1": 2.0})  # cosine distance 1 - 1/sqrt(5) ~= 0.55279
FAR_EMBEDDING_5 = _vector(**{"5": 1.0})  # orthogonal to ANCHOR_EMBEDDING -- cosine distance 1
FAR_EMBEDDING_6 = _vector(**{"6": 1.0})
FAR_EMBEDDING_7 = _vector(**{"7": 1.0})


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real shell/`.env` from leaking into `score_stage`'s
    internal `Settings()` call (#46), same `_isolate_env` pattern
    `tests/test_pipeline_extract.py` uses for `extract_stage`.

    Only `MAX_SCORE_ATTEMPTS` is deleted -- it's the only `Settings`
    field `score_stage` reads -- so a stray `MAX_SCORE_ATTEMPTS` in the
    environment can't desync these tests from the cap value the
    `max_score_attempts` fixture below reads back off `Settings()`.
    """
    monkeypatch.delenv("MAX_SCORE_ATTEMPTS", raising=False)


@pytest.fixture
def max_score_attempts(_isolate_env: None) -> int:
    """The effective `Settings().max_score_attempts` cap for these tests.

    Read off a real `Settings()` instance (with the environment already
    isolated by `_isolate_env`) rather than hardcoded, so these tests stay
    correct against whatever `design.md`-listed default `Settings` defines
    -- currently `3`.
    """
    return Settings().max_score_attempts


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
async def _clear_stale_unscored_events(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Move every pre-existing `relevance IS NULL` `event` row out of
    `score_stage`'s selection.

    `score_stage`'s selection query is global -- `Event.relevance.is_(None)`,
    no `watch_id` filter, by design (per the issue). The Compose Postgres is
    shared and never truncated between test runs, so any event left
    unscored by an earlier test run (or an earlier test in this file that
    doesn't itself call `score_stage`) would otherwise leak into a later
    test's selection -- same global-selection-query test-pollution
    `tests/test_pipeline_adjudicate.py` hits for its own selection query.
    Runs before each test's own event rows are created, so it only ever
    touches pre-existing rows, never the test's own fixtures.
    """
    async with session_factory() as session:
        await session.execute(
            update(Event).where(Event.relevance.is_(None)).values(relevance="irrelevant")
        )
        await session.commit()


async def _make_watch(session: AsyncSession, prefix: str = "score-test") -> Watch:
    watch = Watch(slug=unique_slug(prefix), name="Score Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch


def _make_event(
    watch_id: uuid.UUID,
    *,
    embedding: list[float],
    entities: list[str] | None = None,
    event_date: datetime | None = None,
    title: str = "Some Event",
) -> Event:
    return Event(
        watch_id=watch_id,
        title=title,
        fact_summary="Some fact summary.",
        interpretation="Some interpretation.",
        event_date=event_date,
        entities=entities if entities is not None else [],
        embedding=embedding,
    )


def _make_context_item(watch_id: uuid.UUID, *, kind: str, label: str) -> ContextItem:
    return ContextItem(watch_id=watch_id, kind=kind, label=label, body=f"Body for {label}.")


# ---------------------------------------------------------------------------
# score_stage LLM stub helpers -- same pattern
# `tests/test_pipeline_adjudicate.py` uses: a real `LLMClient` built with
# `Settings(_env_file=None, llm_api_key="test-key", ...)`, with
# `client._client.chat.completions.create` monkeypatched to an `AsyncMock`.
# ---------------------------------------------------------------------------


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


async def _fetch_event(
    session_factory: async_sessionmaker[AsyncSession], event_id: uuid.UUID
) -> Event:
    async with session_factory() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        return result.scalar_one()


# ---------------------------------------------------------------------------
# find_nearest_events (match.py)
# ---------------------------------------------------------------------------


async def test_find_nearest_events_ranks_excludes_self_no_recency_filter(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    now = datetime.now(UTC)
    very_old = now - timedelta(days=400)

    async with session_factory() as session:
        watch = await _make_watch(session)
        other_watch = await _make_watch(session, prefix="score-test-other")

        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        # Identical direction, but a very old event_date -- still included,
        # unlike `find_candidate_events`'s recency-windowed behavior.
        near_old = _make_event(
            watch.id, embedding=DIST_0_EMBEDDING, event_date=very_old, title="Near but old"
        )
        mid = _make_event(watch.id, embedding=DIST_1_EMBEDDING, title="Mid")
        far = _make_event(watch.id, embedding=DIST_2_EMBEDDING, title="Far")
        other_watch_event = _make_event(
            other_watch.id, embedding=DIST_0_EMBEDDING, title="Different watch"
        )
        session.add_all([anchor, near_old, mid, far, other_watch_event])
        await session.commit()
        await session.refresh(anchor)

        candidates = await find_nearest_events(session, anchor)

    assert all(isinstance(candidate, MatchCandidate) for candidate in candidates)
    assert [candidate.event_id for candidate in candidates] == [near_old.id, mid.id, far.id]
    assert candidates[0].distance == pytest.approx(0.0, abs=1e-4)
    assert candidates[1].distance == pytest.approx(1 - 1 / (2**0.5), abs=1e-4)
    assert candidates[2].distance == pytest.approx(1 - 1 / (5**0.5), abs=1e-4)
    returned_ids = {candidate.event_id for candidate in candidates}
    assert anchor.id not in returned_ids
    assert other_watch_event.id not in returned_ids


async def test_find_nearest_events_truncates_to_limit(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        nearest = _make_event(watch.id, embedding=DIST_0_EMBEDDING, title="Nearest")
        middle = _make_event(watch.id, embedding=DIST_1_EMBEDDING, title="Middle")
        farthest = _make_event(watch.id, embedding=DIST_2_EMBEDDING, title="Farthest")
        session.add_all([anchor, nearest, middle, farthest])
        await session.commit()
        await session.refresh(anchor)

        candidates = await find_nearest_events(session, anchor, limit=2)

    assert [candidate.event_id for candidate in candidates] == [nearest.id, middle.id]


# ---------------------------------------------------------------------------
# build_context_bundle -- context_item split
# ---------------------------------------------------------------------------


async def test_build_context_bundle_splits_context_items_by_kind_and_watch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        other_watch = await _make_watch(session, prefix="score-test-other")

        system_item_1 = _make_context_item(watch.id, kind="system", label="Silver background 1")
        system_item_2 = _make_context_item(watch.id, kind="system", label="Silver background 2")
        user_item = _make_context_item(watch.id, kind="user", label="User note")
        other_watch_item = _make_context_item(other_watch.id, kind="system", label="Not ours")
        session.add_all([system_item_1, system_item_2, user_item, other_watch_item])

        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        session.add(anchor)
        await session.commit()
        await session.refresh(anchor)

        bundle = await build_context_bundle(session, watch, anchor, vector_limit=0)

    assert isinstance(bundle, ContextBundle)
    assert {item.id for item in bundle.system_context} == {system_item_1.id, system_item_2.id}
    assert {item.id for item in bundle.user_context} == {user_item.id}
    other_item_ids = {item.id for item in bundle.system_context + bundle.user_context}
    assert other_watch_item.id not in other_item_ids
    assert bundle.related_events == []
    assert bundle.feedback_summary == []


# ---------------------------------------------------------------------------
# build_context_bundle -- related_events merge/dedup/shared_entities
# ---------------------------------------------------------------------------


async def test_build_context_bundle_related_events_merge_dedup_and_shared_entities(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)

        anchor = _make_event(
            watch.id, embedding=ANCHOR_EMBEDDING, entities=["Apple", "Fed"], title="Anchor"
        )
        # Vector-only: nearest, no entity overlap.
        vector_only = _make_event(
            watch.id, embedding=DIST_0_EMBEDDING, entities=["Foo"], title="Vector only"
        )
        # Both relation- and vector-linked: 45 degrees off (still within
        # vector_limit=2), overlaps on "Apple".
        merged = _make_event(
            watch.id,
            embedding=DIST_1_EMBEDDING,
            entities=["Apple", "Foo"],
            title="Merged via relation and vector",
        )
        # Relation-only (anchor is `from_event_id`): far in embedding space,
        # so outside vector_limit=2 -- linked by two relation rows.
        relation_only_from = _make_event(
            watch.id, embedding=FAR_EMBEDDING_5, entities=["Apple"], title="Relation only (from)"
        )
        # Relation-only (anchor is `to_event_id`): the other relation
        # direction still counts as "linked".
        relation_only_to = _make_event(
            watch.id, embedding=FAR_EMBEDDING_6, entities=["Fed", "ECB"], title="Relation only (to)"
        )
        # Unrelated: far in embedding space (outside vector_limit=2) and no
        # relation row -- must not appear in related_events at all.
        unrelated = _make_event(
            watch.id, embedding=FAR_EMBEDDING_7, entities=["Apple"], title="Unrelated"
        )
        session.add_all(
            [anchor, vector_only, merged, relation_only_from, relation_only_to, unrelated]
        )
        await session.commit()
        await session.refresh(anchor)

        session.add_all(
            [
                EventRelation(
                    from_event_id=anchor.id,
                    to_event_id=merged.id,
                    relation="escalation-of",
                    rationale="Merged fixture.",
                ),
                EventRelation(
                    from_event_id=anchor.id,
                    to_event_id=relation_only_from.id,
                    relation="precedes",
                    rationale="First relation row for the same pair.",
                ),
                EventRelation(
                    from_event_id=anchor.id,
                    to_event_id=relation_only_from.id,
                    relation="context-for",
                    rationale="Second relation row for the same pair.",
                ),
                EventRelation(
                    from_event_id=relation_only_to.id,
                    to_event_id=anchor.id,
                    relation="similar",
                    rationale="Anchor is the to_event_id here.",
                ),
            ]
        )
        await session.commit()

        bundle = await build_context_bundle(session, watch, anchor, vector_limit=2)

    related_by_id = {related.event.id: related for related in bundle.related_events}
    assert set(related_by_id.keys()) == {
        vector_only.id,
        merged.id,
        relation_only_from.id,
        relation_only_to.id,
    }
    assert unrelated.id not in related_by_id

    vector_only_related = related_by_id[vector_only.id]
    assert vector_only_related.matched_via == frozenset({"vector"})
    assert vector_only_related.distance == pytest.approx(0.0, abs=1e-4)
    assert vector_only_related.shared_entities == []

    merged_related = related_by_id[merged.id]
    assert merged_related.matched_via == frozenset({"vector", "relation:escalation-of"})
    assert merged_related.distance == pytest.approx(1 - 1 / (2**0.5), abs=1e-4)
    assert merged_related.shared_entities == ["Apple"]

    relation_only_from_related = related_by_id[relation_only_from.id]
    assert relation_only_from_related.matched_via == frozenset(
        {"relation:precedes", "relation:context-for"}
    )
    assert relation_only_from_related.distance is None
    assert relation_only_from_related.shared_entities == ["Apple"]

    relation_only_to_related = related_by_id[relation_only_to.id]
    assert relation_only_to_related.matched_via == frozenset({"relation:similar"})
    assert relation_only_to_related.distance is None
    assert relation_only_to_related.shared_entities == ["Fed"]

    # Deterministic order: vector-ranked first (ascending distance), then
    # relation-only events with no distance.
    assert [related.event.id for related in bundle.related_events[:2]] == [
        vector_only.id,
        merged.id,
    ]
    assert all(isinstance(related, RelatedEvent) for related in bundle.related_events)


async def test_build_context_bundle_related_events_empty_when_no_relations_or_neighbors(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        # The only event for this watch -- zero event_relation rows and
        # nothing for find_nearest_events to find.
        lonely = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Lonely event")
        session.add(lonely)
        await session.commit()
        await session.refresh(lonely)

        bundle = await build_context_bundle(session, watch, lonely)

    assert bundle.related_events == []


# ---------------------------------------------------------------------------
# build_context_bundle -- feedback_summary bucketing
# ---------------------------------------------------------------------------


async def test_build_context_bundle_feedback_summary_buckets_by_category_and_window(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Also covers #57's multi-category note duplication: the `event_x`
    "useful" feedback row below carries a `note`, and `event_x` is linked
    to both `markets` and `policy` -- that note must appear in both
    buckets' `notes`, mirroring `count`'s existing double-count-by-design
    behavior for the same row."""
    now = datetime.now(UTC)
    window_days = 10
    inside_buffer = now - timedelta(days=window_days) + timedelta(minutes=5)
    outside_buffer = now - timedelta(days=window_days) - timedelta(minutes=5)

    async with session_factory() as session:
        watch = await _make_watch(session)
        other_watch = await _make_watch(session, prefix="score-test-other")

        # Reuse two of the real, idempotently-seeded categories rather than
        # inserting new `Category` rows -- see the module docstring.
        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()
        policy_result = await session.execute(select(Category).where(Category.slug == "price"))
        policy = policy_result.scalar_one()

        # event_x belongs to both categories -- a feedback row on it must
        # increment the count in both buckets (double-counts by design).
        event_x = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Event X")
        event_y = _make_event(watch.id, embedding=DIST_0_EMBEDDING, title="Event Y")
        other_watch_event = _make_event(
            other_watch.id, embedding=ANCHOR_EMBEDDING, title="Other watch event"
        )
        session.add_all([event_x, event_y, other_watch_event])
        await session.commit()
        await session.refresh(event_x)
        await session.refresh(event_y)
        await session.refresh(other_watch_event)

        session.add_all(
            [
                EventCategory(event_id=event_x.id, category_id=markets.id),
                EventCategory(event_id=event_x.id, category_id=policy.id),
                EventCategory(event_id=event_y.id, category_id=markets.id),
                EventCategory(event_id=other_watch_event.id, category_id=markets.id),
            ]
        )
        await session.commit()

        session.add_all(
            [
                # Multi-category double-count: one row, two buckets bumped
                # -- and (#57) its note surfaced in both buckets' `notes`.
                Feedback(
                    watch_id=watch.id,
                    event_id=event_x.id,
                    verdict="useful",
                    note="Great catch on the mine strike.",
                    created_at=now - timedelta(days=2),
                ),
                Feedback(
                    watch_id=watch.id,
                    event_id=event_y.id,
                    verdict="useful",
                    created_at=now - timedelta(days=3),
                ),
                Feedback(
                    watch_id=watch.id,
                    event_id=event_y.id,
                    verdict="not_useful",
                    created_at=now - timedelta(days=1),
                ),
                # Just inside the window -- counted.
                Feedback(
                    watch_id=watch.id,
                    event_id=event_x.id,
                    verdict="less_of_this",
                    created_at=inside_buffer,
                ),
                # Just outside the window -- excluded.
                Feedback(
                    watch_id=watch.id,
                    event_id=event_x.id,
                    verdict="less_of_this",
                    created_at=outside_buffer,
                ),
                # Different watch, inside the window -- excluded.
                Feedback(
                    watch_id=other_watch.id,
                    event_id=other_watch_event.id,
                    verdict="useful",
                    created_at=now - timedelta(days=1),
                ),
            ]
        )
        await session.commit()

        bundle = await build_context_bundle(
            session, watch, event_x, vector_limit=0, feedback_window_days=window_days
        )

    assert bundle.feedback_summary == [
        FeedbackBucket(category_slug=markets.slug, verdict="less_of_this", count=1, notes=[]),
        FeedbackBucket(category_slug=markets.slug, verdict="not_useful", count=1, notes=[]),
        FeedbackBucket(
            category_slug=markets.slug,
            verdict="useful",
            count=2,
            notes=["Great catch on the mine strike."],
        ),
        FeedbackBucket(category_slug=policy.slug, verdict="less_of_this", count=1, notes=[]),
        FeedbackBucket(
            category_slug=policy.slug,
            verdict="useful",
            count=1,
            notes=["Great catch on the mine strike."],
        ),
    ]


# ---------------------------------------------------------------------------
# build_context_bundle -- feedback notes (#57)
# ---------------------------------------------------------------------------


async def test_build_context_bundle_feedback_notes_excludes_blank_and_null_notes(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A bucket mixing noted and note-less rows: `count` includes every
    row, but `notes` only ever carries the non-blank ones -- a `None`
    note (the normal `submit_feedback` shape for "no note given") and a
    whitespace-only note (simulating a row written some other way, since
    `submit_feedback` itself already normalizes blank input to `None`)
    both contribute to `count` and both stay out of `notes`."""
    async with session_factory() as session:
        watch = await _make_watch(session)
        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()

        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Noted event")
        session.add(event)
        await session.commit()
        await session.refresh(event)

        session.add(EventCategory(event_id=event.id, category_id=markets.id))
        await session.commit()

        now = datetime.now(UTC)
        session.add_all(
            [
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note="This one has real text.",
                    created_at=now - timedelta(minutes=1),
                ),
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note=None,
                    created_at=now - timedelta(minutes=2),
                ),
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note="   ",
                    created_at=now - timedelta(minutes=3),
                ),
            ]
        )
        await session.commit()

        bundle = await build_context_bundle(session, watch, event, vector_limit=0)

    assert bundle.feedback_summary == [
        FeedbackBucket(
            category_slug=markets.slug,
            verdict="useful",
            count=3,
            notes=["This one has real text."],
        ),
    ]


async def test_build_context_bundle_feedback_notes_caps_at_five_most_recent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A bucket with more than `FEEDBACK_NOTES_PER_BUCKET` (5) noted rows
    only carries its 5 newest, ordered `created_at` descending; `count`
    still reflects all of them."""
    async with session_factory() as session:
        watch = await _make_watch(session)
        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()

        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Popular event")
        session.add(event)
        await session.commit()
        await session.refresh(event)

        session.add(EventCategory(event_id=event.id, category_id=markets.id))
        await session.commit()

        now = datetime.now(UTC)
        # 7 noted rows, newest first: note-0 is most recent, note-6 oldest.
        session.add_all(
            [
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note=f"note-{i}",
                    created_at=now - timedelta(minutes=i),
                )
                for i in range(7)
            ]
        )
        await session.commit()

        bundle = await build_context_bundle(session, watch, event, vector_limit=0)

    assert bundle.feedback_summary == [
        FeedbackBucket(
            category_slug=markets.slug,
            verdict="useful",
            count=7,
            notes=["note-0", "note-1", "note-2", "note-3", "note-4"],
        ),
    ]


async def test_build_context_bundle_feedback_notes_dedup_exact_match_within_bucket(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A bucket with 40 noted rows in the window where only 3 are
    text-distinct after `.strip().casefold()` normalization ends up with
    exactly those 3 notes, not 5 slots padded with repeats (#61). The 37
    duplicate rows vary only in case and surrounding whitespace -- proving
    normalization, not plain `==`, drives the dedup -- and are all older
    than the 3 canonical rows, so the loop must keep scanning past every
    duplicate to find enough distinct notes."""
    async with session_factory() as session:
        watch = await _make_watch(session)
        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()

        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Repetitive event")
        session.add(event)
        await session.commit()
        await session.refresh(event)

        session.add(EventCategory(event_id=event.id, category_id=markets.id))
        await session.commit()

        now = datetime.now(UTC)
        canonical_texts = [
            "Too many similar stories about this event.",
            "This alert did not match my portfolio at all.",
            "Exactly the kind of catch i wanted, nice work.",
        ]
        # Case/whitespace-only variants of each canonical text -- all
        # normalize to the same `.strip().casefold()` value as their
        # canonical original.
        variants = [
            lambda s: s.upper(),
            lambda s: f"  {s}  ",
            lambda s: s.swapcase(),
            lambda s: f"\t{s.upper()}\t",
        ]

        rows = [
            # The 3 canonical rows are the most recent -- minutes 0, 1, 2.
            Feedback(
                watch_id=watch.id,
                event_id=event.id,
                verdict="useful",
                note=text,
                created_at=now - timedelta(minutes=index),
            )
            for index, text in enumerate(canonical_texts)
        ]
        # 37 older duplicate rows (minutes 3..39), cycling through the 3
        # canonical texts and the 4 case/whitespace transforms.
        for offset in range(37):
            canonical = canonical_texts[offset % len(canonical_texts)]
            transform = variants[offset % len(variants)]
            rows.append(
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note=transform(canonical),
                    created_at=now - timedelta(minutes=3 + offset),
                )
            )
        assert len(rows) == 40
        session.add_all(rows)
        await session.commit()

        bundle = await build_context_bundle(session, watch, event, vector_limit=0)

    assert bundle.feedback_summary == [
        FeedbackBucket(
            category_slug=markets.slug,
            verdict="useful",
            count=40,
            notes=canonical_texts,
        ),
    ]


async def test_build_context_bundle_feedback_notes_near_duplicates_stay_distinct(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Exact-match dedup (#61) is not fuzzy/semantic: two notes differing
    only in punctuation or wording (not just case/whitespace) both survive
    as distinct entries, even alongside a genuine case/whitespace
    duplicate that does get skipped."""
    async with session_factory() as session:
        watch = await _make_watch(session)
        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()

        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Near-duplicate event")
        session.add(event)
        await session.commit()
        await session.refresh(event)

        session.add(EventCategory(event_id=event.id, category_id=markets.id))
        await session.commit()

        now = datetime.now(UTC)
        original = "Great catch, exactly what I wanted!"
        case_whitespace_duplicate = f"  {original.upper()}  "
        punctuation_variant = "Great catch, exactly what I wanted."  # "." not "!"
        session.add_all(
            [
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note=original,
                    created_at=now - timedelta(minutes=1),
                ),
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note=case_whitespace_duplicate,
                    created_at=now - timedelta(minutes=2),
                ),
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note=punctuation_variant,
                    created_at=now - timedelta(minutes=3),
                ),
            ]
        )
        await session.commit()

        bundle = await build_context_bundle(session, watch, event, vector_limit=0)

    assert bundle.feedback_summary == [
        FeedbackBucket(
            category_slug=markets.slug,
            verdict="useful",
            count=3,
            notes=[original, punctuation_variant],
        ),
    ]


async def test_build_context_bundle_feedback_notes_truncated_over_char_limit(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A note longer than `FEEDBACK_NOTE_CHAR_LIMIT` (200) chars is cut to
    the first 200 with a trailing "..." marker; a note at or under the
    limit is left verbatim, with no marker appended."""
    async with session_factory() as session:
        watch = await _make_watch(session)
        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()

        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Verbose feedback event")
        session.add(event)
        await session.commit()
        await session.refresh(event)

        session.add(EventCategory(event_id=event.id, category_id=markets.id))
        await session.commit()

        now = datetime.now(UTC)
        long_note = "x" * 250
        exact_note = "y" * 200
        session.add_all(
            [
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="useful",
                    note=long_note,
                    created_at=now - timedelta(minutes=1),
                ),
                Feedback(
                    watch_id=watch.id,
                    event_id=event.id,
                    verdict="not_useful",
                    note=exact_note,
                    created_at=now - timedelta(minutes=1),
                ),
            ]
        )
        await session.commit()

        bundle = await build_context_bundle(session, watch, event, vector_limit=0)

    bucket_by_verdict = {bucket.verdict: bucket for bucket in bundle.feedback_summary}
    assert bucket_by_verdict["useful"].notes == [("x" * 200) + "..."]
    assert bucket_by_verdict["not_useful"].notes == [exact_note]


async def test_build_context_bundle_feedback_note_disappears_after_withdrawal(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A withdrawn (hard-deleted, #51) `Feedback` row's note no longer
    appears on the next `build_context_bundle` call -- no special-casing
    needed, since the query reads live table state."""
    async with session_factory() as session:
        watch = await _make_watch(session)
        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()

        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Withdrawable event")
        session.add(event)
        await session.commit()
        await session.refresh(event)

        session.add(EventCategory(event_id=event.id, category_id=markets.id))
        await session.commit()

        feedback = Feedback(
            watch_id=watch.id,
            event_id=event.id,
            verdict="useful",
            note="I will withdraw this.",
            created_at=datetime.now(UTC),
        )
        session.add(feedback)
        await session.commit()

        bundle_before = await build_context_bundle(session, watch, event, vector_limit=0)
        assert bundle_before.feedback_summary == [
            FeedbackBucket(
                category_slug=markets.slug,
                verdict="useful",
                count=1,
                notes=["I will withdraw this."],
            ),
        ]

        # Hard-delete, same effect as `withdraw_feedback` (#51).
        await session.delete(feedback)
        await session.commit()

        bundle_after = await build_context_bundle(session, watch, event, vector_limit=0)

    assert bundle_after.feedback_summary == []


# ---------------------------------------------------------------------------
# score_stage (#28)
# ---------------------------------------------------------------------------


async def test_score_stage_persists_irrelevant_verdict(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "irrelevant", "importance": "low", '
        '"impact_direction": "neutral", "impact_reason": "Not related to silver.", '
        '"impact_confidence": "low"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Unrelated news")
        session.add(event)
        await session.commit()
        event_id = event.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 1, "scored": 0, "skipped": 0, "score_capped": 0}

    row = await _fetch_event(session_factory, event_id)
    assert row.relevance == "irrelevant"
    assert row.importance == "low"
    assert row.impact_direction == "neutral"
    assert row.impact_reason == "Not related to silver."
    assert row.impact_confidence == "low"


async def test_score_stage_persists_high_relevance_verdict(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "high", "importance": "critical", '
        '"impact_direction": "bullish", "impact_reason": "Major supply disruption.", '
        '"impact_confidence": "high"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Silver mine strike")
        session.add(event)
        await session.commit()
        event_id = event.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 1, "skipped": 0, "score_capped": 0}

    row = await _fetch_event(session_factory, event_id)
    assert row.relevance == "high"
    assert row.importance == "critical"
    assert row.impact_direction == "bullish"
    assert row.impact_reason == "Major supply disruption."
    assert row.impact_confidence == "high"


async def test_score_stage_persists_unclear_impact_direction(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """`"unclear"` is a valid, not a fallback, `impact_direction` value."""
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "medium", "importance": "medium", '
        '"impact_direction": "unclear", "impact_reason": "Conflicting signals.", '
        '"impact_confidence": "medium"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Mixed signals report")
        session.add(event)
        await session.commit()
        event_id = event.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 1, "skipped": 0, "score_capped": 0}

    row = await _fetch_event(session_factory, event_id)
    assert row.relevance == "medium"
    assert row.importance == "medium"
    assert row.impact_direction == "unclear"
    assert row.impact_reason == "Conflicting signals."
    assert row.impact_confidence == "medium"


async def test_score_stage_skips_row_malformed_on_both_attempts(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    # Structurally valid JSON but violates the `impact_reason` min_length
    # constraint on both `call_structured` attempts -> `ValidationError`
    # both times.
    stub_create.side_effect = [
        _FakeChatCompletion(
            '{"relevance": "high", "importance": "high", '
            '"impact_direction": "bullish", "impact_reason": "", '
            '"impact_confidence": "high"}'
        ),
        _FakeChatCompletion(
            '{"relevance": "high", "importance": "high", '
            '"impact_direction": "bullish", "impact_reason": "", '
            '"impact_confidence": "high"}'
        ),
    ]

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Bad response event")
        session.add(event)
        await session.commit()
        event_id = event.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 0, "skipped": 1, "score_capped": 0}
    assert stub_create.call_count == 2

    row = await _fetch_event(session_factory, event_id)
    assert row.relevance is None
    assert row.importance is None
    assert row.impact_direction is None
    assert row.impact_reason is None
    assert row.impact_confidence is None


async def test_score_stage_selection_skips_already_scored_events(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "medium", "importance": "medium", '
        '"impact_direction": "neutral", "impact_reason": "Some reason.", '
        '"impact_confidence": "medium"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        already_scored = _make_event(
            watch.id, embedding=ANCHOR_EMBEDDING, title="Already scored event"
        )
        already_scored.relevance = "low"
        already_scored.importance = "low"
        already_scored.impact_direction = "neutral"
        already_scored.impact_reason = "Existing reason, untouched."
        already_scored.impact_confidence = "low"
        unscored = _make_event(
            watch.id, embedding=DIST_0_EMBEDDING, title="Not yet scored event"
        )
        session.add_all([already_scored, unscored])
        await session.commit()
        already_scored_id = already_scored.id
        unscored_id = unscored.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 1, "skipped": 0, "score_capped": 0}
    assert stub_create.call_count == 1

    already_scored_row = await _fetch_event(session_factory, already_scored_id)
    assert already_scored_row.relevance == "low"
    assert already_scored_row.importance == "low"
    assert already_scored_row.impact_direction == "neutral"
    assert already_scored_row.impact_reason == "Existing reason, untouched."
    assert already_scored_row.impact_confidence == "low"

    unscored_row = await _fetch_event(session_factory, unscored_id)
    assert unscored_row.relevance == "medium"
    assert unscored_row.importance == "medium"


async def test_score_stage_renders_full_context_bundle_in_prompt(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A real prompt-content check: builds a `ContextBundle`-producing
    fixture (system/user context items, a relation-linked related event,
    and a feedback row) and asserts the rendered prompt sent to the stub
    actually contains that content -- not just asserted in prose."""
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "medium", "importance": "medium", '
        '"impact_direction": "neutral", "impact_reason": "Some reason.", '
        '"impact_confidence": "medium"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)

        system_item = _make_context_item(
            watch.id, kind="system", label="Silver Background Fact"
        )
        user_item = _make_context_item(watch.id, kind="user", label="User Watch Note")
        session.add_all([system_item, user_item])

        anchor = _make_event(
            watch.id,
            embedding=ANCHOR_EMBEDDING,
            entities=["Apple", "Fed"],
            title="Anchor event to be scored",
        )
        related = _make_event(
            watch.id,
            embedding=FAR_EMBEDDING_5,
            entities=["Apple"],
            title="Related historical event",
        )
        # Already scored -- must not itself be picked up by score_stage's
        # selection; only `anchor` should be scored in this test.
        related.relevance = "medium"
        session.add_all([anchor, related])
        await session.commit()
        await session.refresh(anchor)
        await session.refresh(related)

        session.add(
            EventRelation(
                from_event_id=anchor.id,
                to_event_id=related.id,
                relation="precedes",
                rationale="Fixture relation.",
            )
        )

        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()
        session.add(EventCategory(event_id=anchor.id, category_id=markets.id))
        await session.commit()

        session.add(
            Feedback(
                watch_id=watch.id,
                event_id=anchor.id,
                verdict="useful",
                note="Great catch on the timing of this one.",
                created_at=datetime.now(UTC),
            )
        )
        await session.commit()
        anchor_id = anchor.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 1, "skipped": 0, "score_capped": 0}

    prompt_content = stub_create.call_args.kwargs["messages"][0]["content"]

    # System/user context items.
    assert system_item.label in prompt_content
    assert system_item.body in prompt_content
    assert user_item.label in prompt_content
    assert user_item.body in prompt_content

    # Related event content -- title, matched_via tag, shared entity.
    assert "Related historical event" in prompt_content
    assert "relation:precedes" in prompt_content
    assert "Apple" in prompt_content

    # Feedback summary line, plus (#57) its note text literally present
    # in the rendered message sent to the stub client -- not just on the
    # ContextBundle object.
    assert "market: 1 x 'useful'" in prompt_content
    assert "Great catch on the timing of this one." in prompt_content

    # The event itself, last.
    assert "Anchor event to be scored" in prompt_content

    assert anchor_id is not None


async def test_score_stage_feedback_summary_count_only_line_unchanged_without_notes(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """(#57) A bucket with zero noted rows keeps its exact
    `"<slug>: <count> x '<verdict>'"` count-only line, with no `notes:`
    sub-line beneath it -- even while a sibling bucket (for a different
    category) that does have a note gets one."""
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "medium", "importance": "medium", '
        '"impact_direction": "neutral", "impact_reason": "Some reason.", '
        '"impact_confidence": "medium"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        await seed_categories(session)
        markets_result = await session.execute(select(Category).where(Category.slug == "market"))
        markets = markets_result.scalar_one()
        price_result = await session.execute(select(Category).where(Category.slug == "price"))
        price = price_result.scalar_one()

        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor to be scored")
        noteless_event = _make_event(
            watch.id, embedding=DIST_0_EMBEDDING, title="Noteless feedback event"
        )
        noted_event = _make_event(
            watch.id, embedding=DIST_1_EMBEDDING, title="Noted feedback event"
        )
        noteless_event.relevance = "medium"
        noted_event.relevance = "medium"
        session.add_all([anchor, noteless_event, noted_event])
        await session.commit()
        await session.refresh(noteless_event)
        await session.refresh(noted_event)

        session.add_all(
            [
                EventCategory(event_id=noteless_event.id, category_id=markets.id),
                EventCategory(event_id=noted_event.id, category_id=price.id),
            ]
        )
        await session.commit()

        session.add_all(
            [
                Feedback(
                    watch_id=watch.id,
                    event_id=noteless_event.id,
                    verdict="useful",
                    created_at=datetime.now(UTC),
                ),
                Feedback(
                    watch_id=watch.id,
                    event_id=noted_event.id,
                    verdict="useful",
                    note="Explains why this was useful.",
                    created_at=datetime.now(UTC),
                ),
            ]
        )
        await session.commit()

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 1, "skipped": 0, "score_capped": 0}

    prompt_content = stub_create.call_args.kwargs["messages"][0]["content"]
    prompt_lines = prompt_content.splitlines()

    markets_line_index = prompt_lines.index(f"{markets.slug}: 1 x 'useful'")
    assert not prompt_lines[markets_line_index + 1].strip().startswith("notes:")

    price_line_index = prompt_lines.index(f"{price.slug}: 1 x 'useful'")
    assert prompt_lines[price_line_index + 1].strip() == (
        'notes: "Explains why this was useful."'
    )


async def test_score_stage_does_not_touch_fact_summary_or_interpretation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Regression guard: `score_stage` reads `fact_summary`/
    `interpretation` for the prompt only, never writes them -- even with a
    stub response that doesn't correspond to the seeded values."""
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "high", "importance": "critical", '
        '"impact_direction": "bearish", "impact_reason": "Unrelated stub reason.", '
        '"impact_confidence": "high"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Regression event")
        original_fact_summary = event.fact_summary
        original_interpretation = event.interpretation
        session.add(event)
        await session.commit()
        event_id = event.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 1, "skipped": 0, "score_capped": 0}

    row = await _fetch_event(session_factory, event_id)
    assert row.fact_summary == original_fact_summary
    assert row.interpretation == original_interpretation


# ---------------------------------------------------------------------------
# score_attempts cap (#46)
# ---------------------------------------------------------------------------


async def test_score_stage_retries_event_after_one_failed_attempt(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """(#46) An event that fails validation once (`score_attempts == 1`
    after the call, still `relevance IS NULL`) is still selected and
    retried on a second `score_stage` call."""
    client, stub_create = _client_with_stubbed_create()
    bad_response = _FakeChatCompletion(
        '{"relevance": "high", "importance": "high", '
        '"impact_direction": "bullish", "impact_reason": "", '
        '"impact_confidence": "high"}'
    )
    # Structurally valid JSON but violates `impact_reason`'s min_length on
    # both of `call_structured`'s internal attempts -> `ValidationError`
    # propagates, same fixture shape as
    # `test_score_stage_skips_row_malformed_on_both_attempts`.
    stub_create.side_effect = [bad_response, bad_response]

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Retry-once event")
        session.add(event)
        await session.commit()
        event_id = event.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 0, "skipped": 1, "score_capped": 0}
    assert stub_create.call_count == 2

    row = await _fetch_event(session_factory, event_id)
    assert row.relevance is None
    assert row.score_attempts == 1

    # Second score_stage call: still selected (score_attempts=1 is below
    # the default cap of 3), and this time succeeds.
    stub_create.side_effect = None
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "medium", "importance": "medium", '
        '"impact_direction": "neutral", "impact_reason": "Valid on retry.", '
        '"impact_confidence": "medium"}'
    )

    async with session_factory() as session:
        second_result = await score_stage(session, client=client)
        await session.commit()

    assert second_result == {"irrelevant": 0, "scored": 1, "skipped": 0, "score_capped": 0}
    assert stub_create.call_count == 3

    row = await _fetch_event(session_factory, event_id)
    assert row.relevance == "medium"
    assert row.score_attempts == 2


async def test_score_stage_skips_event_already_at_cap(
    session_factory: async_sessionmaker[AsyncSession],
    max_score_attempts: int,
) -> None:
    """(#46) An event already at `score_attempts == max_score_attempts`
    is excluded by the selection query entirely: `call_structured` is
    never called for it, its `score_attempts` is left unchanged, and it's
    counted under `score_capped`, not `skipped`."""
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        '{"relevance": "medium", "importance": "medium", '
        '"impact_direction": "neutral", "impact_reason": "Should never be sent.", '
        '"impact_confidence": "medium"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Capped event")
        event.score_attempts = max_score_attempts
        session.add(event)
        await session.commit()
        event_id = event.id

        result = await score_stage(session, client=client)
        await session.commit()

    assert result == {"irrelevant": 0, "scored": 0, "skipped": 0, "score_capped": 1}
    assert stub_create.call_count == 0

    row = await _fetch_event(session_factory, event_id)
    assert row.relevance is None
    assert row.score_attempts == max_score_attempts


async def test_score_stage_excludes_event_after_reaching_cap_boundary(
    session_factory: async_sessionmaker[AsyncSession],
    max_score_attempts: int,
) -> None:
    """Boundary case (#46): an event that fails validation exactly
    `max_score_attempts` times across that many separate `score_stage`
    calls ends the sequence with `score_attempts == max_score_attempts`
    and `relevance IS NULL`, still having been retried on every one of
    those calls -- then is excluded starting on the *next* call, the
    transition from "still retried" to "capped," not just the two
    steady-state cases the other cap tests cover. Same boundary-test
    precedent #45 established for `extract_stage`."""
    client, stub_create = _client_with_stubbed_create()
    bad_response = _FakeChatCompletion(
        '{"relevance": "high", "importance": "high", '
        '"impact_direction": "bullish", "impact_reason": "", '
        '"impact_confidence": "high"}'
    )

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Boundary event")
        session.add(event)
        await session.commit()
        event_id = event.id

    for attempt in range(1, max_score_attempts + 1):
        stub_create.side_effect = [bad_response, bad_response]
        async with session_factory() as session:
            result = await score_stage(session, client=client)
            await session.commit()
        assert result == {"irrelevant": 0, "scored": 0, "skipped": 1, "score_capped": 0}

        row = await _fetch_event(session_factory, event_id)
        assert row.score_attempts == attempt
        assert row.relevance is None

    # One call past the cap: the row is now excluded rather than retried --
    # no new call to call_structured for it, and it's counted under
    # score_capped this time.
    stub_create.reset_mock(side_effect=True)
    async with session_factory() as session:
        next_result = await score_stage(session, client=client)
        await session.commit()

    assert next_result == {"irrelevant": 0, "scored": 0, "skipped": 0, "score_capped": 1}
    assert stub_create.call_count == 0

    row = await _fetch_event(session_factory, event_id)
    assert row.score_attempts == max_score_attempts
    assert row.relevance is None
