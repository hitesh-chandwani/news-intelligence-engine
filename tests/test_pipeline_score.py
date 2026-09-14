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
SQL/filtering/merge/dedup logic, not embedding quality. `Category` rows
are hand-rolled fixture rows (not the real seeded taxonomy) -- this
module doesn't need the 13 real slugs, only that the `event_category`
join groups correctly, same "small hand-authored fixture" precedent
`_docs/testing-guidelines.md` recommends.

Each test creates its own `Watch`/`Category` rows with `unique_slug`
(same helper `tests/test_models.py`/`tests/test_pipeline_match.py`
define), keeping tests independent of each other and of any prior run's
leftover rows (the Compose Postgres is never truncated between test
runs).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Category, ContextItem, Event, EventCategory, EventRelation, Feedback, Watch
from nie.pipeline.match import MatchCandidate, find_nearest_events
from nie.pipeline.score import ContextBundle, FeedbackBucket, RelatedEvent, build_context_bundle

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
    now = datetime.now(UTC)
    window_days = 10
    inside_buffer = now - timedelta(days=window_days) + timedelta(minutes=5)
    outside_buffer = now - timedelta(days=window_days) - timedelta(minutes=5)

    async with session_factory() as session:
        watch = await _make_watch(session)
        other_watch = await _make_watch(session, prefix="score-test-other")

        markets = Category(slug=unique_slug("markets"), name="Markets")
        policy = Category(slug=unique_slug("policy"), name="Policy")
        session.add_all([markets, policy])

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
                # Multi-category double-count: one row, two buckets bumped.
                Feedback(
                    watch_id=watch.id,
                    event_id=event_x.id,
                    verdict="useful",
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
        FeedbackBucket(category_slug=markets.slug, verdict="less_of_this", count=1),
        FeedbackBucket(category_slug=markets.slug, verdict="not_useful", count=1),
        FeedbackBucket(category_slug=markets.slug, verdict="useful", count=2),
        FeedbackBucket(category_slug=policy.slug, verdict="less_of_this", count=1),
        FeedbackBucket(category_slug=policy.slug, verdict="useful", count=1),
    ]
