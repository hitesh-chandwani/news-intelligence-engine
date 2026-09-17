"""Tests for src/nie/pipeline/relate.py (issue #29).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_pipeline_adjudicate.py`'s/
`tests/test_pipeline_score.py`'s `migrated_db`/`session_factory` fixture
pattern -- a fresh engine per test, not the module-level singleton, so
pooled asyncpg connections stay bound to this test's own event loop.

The LLM is stubbed exactly like `tests/test_pipeline_adjudicate.py`/
`tests/test_pipeline_score.py`: a real `LLMClient` is built with
`Settings(_env_file=None, llm_api_key="test-key", ...)` (no real network
call, no real API key), with `client._client.chat.completions.create`
monkeypatched to an `AsyncMock`, and that client is passed into
`relate_stage(session, client=...)` explicitly.

`relate_stage`'s selection query (`Event.relevance.isnot(None)` AND no
existing outbound `event_relation` row) is global -- no `watch_id`
filter, by design, same as `adjudicate_stage`/`score_stage`. The Compose
Postgres is shared and never truncated between test runs, so an
autouse fixture moves every pre-existing matching row out of selection
before each test, same `_clear_stale_*` precedent those two files set --
here by setting `relevance = NULL` (simplest way to fall out of the
`isnot(None)` half of the selection criterion without needing a valid
`to_event_id` for a dummy relation row).

Every embedding used is a hand-authored fixed 384-length float list, same
precedent as `tests/test_pipeline_match.py`/`tests/test_pipeline_score.py`
-- no live network call, no real embedding model.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
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
from nie.models import Event, EventRelation, Watch
from nie.pipeline.relate import relate_stage

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


# The anchor's embedding: a unit vector along axis 0.
ANCHOR_EMBEDDING = _vector(**{"0": 1.0})
# Identical direction -- cosine distance 0, so nearest via find_nearest_events.
NEAR_EMBEDDING = _vector(**{"0": 1.0})
# Orthogonal -- far in embedding space, outside RELATE_VECTOR_LIMIT's reach
# in practice for these small fixtures, but still reachable by entity
# overlap alone.
FAR_EMBEDDING = _vector(**{"9": 1.0})


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
async def _clear_stale_scored_unrelated_events(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Move every pre-existing `relate_stage`-selectable `event` row out of
    selection.

    `relate_stage`'s selection query is global -- `Event.relevance.isnot(None)`
    AND `Event.related_at.is_(None)` (#47), no `watch_id` filter, by design
    (per the issue). The Compose Postgres is shared and never truncated
    between test runs, so a stray scored-but-unrelated event left by an
    earlier test run (this file's own, or `tests/test_pipeline_score.py`'s)
    would otherwise leak into a later test's selection -- same
    global-selection-query test-pollution `tests/test_pipeline_adjudicate.py`/
    `tests/test_pipeline_score.py` each hit for their own selection queries.
    Setting `relevance = NULL` is the simplest way to fall out of the
    `isnot(None)` half of the criterion, without needing to touch
    `related_at` at all. Runs before each test's own event rows are
    created, so it only ever touches pre-existing rows, never the test's
    own fixtures.
    """
    async with session_factory() as session:
        await session.execute(
            update(Event)
            .where(
                Event.relevance.isnot(None),
                Event.related_at.is_(None),
            )
            .values(relevance=None)
        )
        await session.commit()


async def _make_watch(session: AsyncSession, prefix: str = "relate-test") -> Watch:
    watch = Watch(slug=unique_slug(prefix), name="Relate Test Watch", status="enabled")
    session.add(watch)
    await session.commit()
    return watch


def _make_event(
    watch_id: uuid.UUID,
    *,
    embedding: list[float],
    entities: list[str] | None = None,
    relevance: str | None = "high",
    title: str = "Some Event",
) -> Event:
    return Event(
        watch_id=watch_id,
        title=title,
        fact_summary="Some fact summary.",
        interpretation="Some interpretation.",
        entities=entities if entities is not None else [],
        embedding=embedding,
        relevance=relevance,
    )


# ---------------------------------------------------------------------------
# LLM stub helpers -- same pattern `tests/test_pipeline_adjudicate.py`/
# `tests/test_pipeline_score.py` use.
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


async def _fetch_relations(
    session_factory: async_sessionmaker[AsyncSession], from_event_id: uuid.UUID
) -> list[EventRelation]:
    async with session_factory() as session:
        result = await session.execute(
            select(EventRelation).where(EventRelation.from_event_id == from_event_id)
        )
        return list(result.scalars().all())


async def _fetch_event(
    session_factory: async_sessionmaker[AsyncSession], event_id: uuid.UUID
) -> Event:
    async with session_factory() as session:
        result = await session.execute(select(Event).where(Event.id == event_id))
        return result.scalar_one()


# ---------------------------------------------------------------------------
# relate_stage
# ---------------------------------------------------------------------------


async def test_relate_stage_happy_path_both_candidate_sources(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Two seeded candidates -- one reachable only via vector-nearest, one
    only via entity overlap -- both get persisted `event_relation` rows
    from a stubbed two-relation `RelationSet`."""
    async with session_factory() as session:
        watch = await _make_watch(session)

        anchor = _make_event(
            watch.id, embedding=ANCHOR_EMBEDDING, entities=["Fed"], title="Anchor"
        )
        # Vector-nearest only: identical embedding direction, no shared
        # entity. `relevance=None` -- a historical candidate event, not
        # itself under test for this call's selection (relate_stage's
        # selection is global by Event.relevance, not scoped to `watch`).
        vector_candidate = _make_event(
            watch.id,
            embedding=NEAR_EMBEDDING,
            entities=["Other"],
            relevance=None,
            title="Vector Candidate",
        )
        # Entity-overlap only: far embedding, shares "Fed".
        entity_candidate = _make_event(
            watch.id,
            embedding=FAR_EMBEDDING,
            entities=["Fed"],
            relevance=None,
            title="Entity Candidate",
        )
        session.add_all([anchor, vector_candidate, entity_candidate])
        await session.commit()
        anchor_id = anchor.id
        vector_candidate_id = vector_candidate.id
        entity_candidate_id = entity_candidate.id

    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        f'{{"relations": ['
        f'{{"event_id": "{vector_candidate_id}", "relation": "similar", '
        f'"rationale": "Same development."}}, '
        f'{{"event_id": "{entity_candidate_id}", "relation": "context-for", '
        f'"rationale": "Shares the Fed entity."}}'
        f"]}}"
    )

    async with session_factory() as session:
        result = await relate_stage(session, client=client)
        await session.commit()

    assert result == {"related": 1, "skipped": 0}

    relations = await _fetch_relations(session_factory, anchor_id)
    by_to_id = {relation.to_event_id: relation for relation in relations}
    assert set(by_to_id.keys()) == {vector_candidate_id, entity_candidate_id}

    assert by_to_id[vector_candidate_id].from_event_id == anchor_id
    assert by_to_id[vector_candidate_id].relation == "similar"
    assert by_to_id[vector_candidate_id].rationale == "Same development."

    assert by_to_id[entity_candidate_id].from_event_id == anchor_id
    assert by_to_id[entity_candidate_id].relation == "context-for"
    assert by_to_id[entity_candidate_id].rationale == "Shares the Fed entity."


async def test_relate_stage_drops_hallucinated_event_id_keeps_valid_relation(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        candidate = _make_event(
            watch.id, embedding=NEAR_EMBEDDING, relevance=None, title="Valid Candidate"
        )
        session.add_all([anchor, candidate])
        await session.commit()
        anchor_id = anchor.id
        candidate_id = candidate.id

    hallucinated_id = uuid.uuid4()
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        f'{{"relations": ['
        f'{{"event_id": "{hallucinated_id}", "relation": "similar", '
        f'"rationale": "Invented candidate."}}, '
        f'{{"event_id": "{candidate_id}", "relation": "precedes", '
        f'"rationale": "Real candidate."}}'
        f"]}}"
    )

    async with session_factory() as session:
        result = await relate_stage(session, client=client)
        await session.commit()

    assert result == {"related": 1, "skipped": 0}

    relations = await _fetch_relations(session_factory, anchor_id)
    assert len(relations) == 1
    assert relations[0].to_event_id == candidate_id
    assert relations[0].relation == "precedes"


async def test_relate_stage_drops_self_relation_proposal(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        session.add(anchor)
        await session.commit()
        anchor_id = anchor.id

    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        f'{{"relations": [{{"event_id": "{anchor_id}", "relation": "similar", '
        f'"rationale": "Self-relation, should be dropped."}}]}}'
    )

    async with session_factory() as session:
        result = await relate_stage(session, client=client)
        await session.commit()

    # Zero surviving relations -- counts toward neither "related" nor
    # "skipped" (a legitimately-empty-after-filtering outcome, not an error).
    assert result == {"related": 0, "skipped": 0}

    relations = await _fetch_relations(session_factory, anchor_id)
    assert relations == []


async def test_relate_stage_deduplicates_repeated_pair_within_one_response(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        candidate = _make_event(
            watch.id, embedding=NEAR_EMBEDDING, relevance=None, title="Candidate"
        )
        session.add_all([anchor, candidate])
        await session.commit()
        anchor_id = anchor.id
        candidate_id = candidate.id

    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        f'{{"relations": ['
        f'{{"event_id": "{candidate_id}", "relation": "similar", "rationale": "First."}}, '
        f'{{"event_id": "{candidate_id}", "relation": "similar", "rationale": "Duplicate."}}'
        f"]}}"
    )

    async with session_factory() as session:
        result = await relate_stage(session, client=client)
        await session.commit()

    assert result == {"related": 1, "skipped": 0}

    relations = await _fetch_relations(session_factory, anchor_id)
    assert len(relations) == 1
    assert relations[0].to_event_id == candidate_id
    assert relations[0].relation == "similar"
    assert relations[0].rationale == "First."


async def test_relate_stage_does_not_duplicate_pair_already_persisted(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A `(to_event_id, relation)` pair already persisted for `from_event_id`
    is never re-proposed to the LLM in the first place -- `relate_stage`'s
    own selection clause (`Event.related_at.is_(None)`, #47) excludes any
    event that already completed a `relate_stage` pass from every future
    selection, so the in-loop "already persisted" filter this guards
    against (see `relate_stage`'s docstring) can only ever see an empty
    `persisted_pairs` set for an event it processes -- it is unreachable
    via `relate_stage`'s own selection gate, same "defense in depth, but
    assert it explicitly rather than relying on the exclusion silently
    holding" precedent the issue sets for the self-relation check. This
    test pins down the resulting observable guarantee: an event with a
    pre-existing persisted relation (and, per #47, the `related_at` that
    same earlier pass would have set alongside it) is never re-selected,
    never re-sent to the LLM, and its existing row is left untouched -- so
    no duplicate and no `IntegrityError` can ever occur for it, regardless
    of what any stub response would propose."""
    async with session_factory() as session:
        watch = await _make_watch(session)
        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        candidate = _make_event(
            watch.id, embedding=NEAR_EMBEDDING, relevance=None, title="Candidate"
        )
        session.add_all([anchor, candidate])
        await session.commit()
        anchor_id = anchor.id
        candidate_id = candidate.id

        # `related_at` is set alongside the relation here because a real
        # earlier `relate_stage` pass would have set both together (#47) --
        # this fixture is standing in for that already-completed pass.
        anchor.related_at = datetime.now(UTC)
        session.add(
            EventRelation(
                from_event_id=anchor_id,
                to_event_id=candidate_id,
                relation="similar",
                rationale="Already persisted from an earlier run.",
            )
        )
        await session.commit()

    client, stub_create = _client_with_stubbed_create()
    # Even if the LLM were called and proposed this exact pair again, no
    # duplicate could result -- but it must not be called for `anchor` at
    # all, since `anchor` is not selected.
    stub_create.return_value = _FakeChatCompletion(
        f'{{"relations": [{{"event_id": "{candidate_id}", "relation": "similar", '
        f'"rationale": "Proposed again, duplicates the persisted pair."}}]}}'
    )

    async with session_factory() as session:
        result = await relate_stage(session, client=client)
        await session.commit()

    assert result == {"related": 0, "skipped": 0}
    assert stub_create.call_count == 0

    relations = await _fetch_relations(session_factory, anchor_id)
    assert len(relations) == 1
    assert relations[0].rationale == "Already persisted from an earlier run."


async def test_relate_stage_is_idempotent_across_two_calls(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        watch = await _make_watch(session)
        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        candidate = _make_event(
            watch.id, embedding=NEAR_EMBEDDING, relevance=None, title="Candidate"
        )
        session.add_all([anchor, candidate])
        await session.commit()
        anchor_id = anchor.id
        candidate_id = candidate.id

    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion(
        f'{{"relations": [{{"event_id": "{candidate_id}", "relation": "similar", '
        f'"rationale": "First call."}}]}}'
    )

    async with session_factory() as session:
        first_result = await relate_stage(session, client=client)
        await session.commit()

    assert first_result == {"related": 1, "skipped": 0}
    assert stub_create.call_count == 1

    async with session_factory() as session:
        second_result = await relate_stage(session, client=client)
        await session.commit()

    assert second_result == {"related": 0, "skipped": 0}
    # `anchor` already has `related_at` set from the first call -- the
    # `Event.related_at.is_(None)` selection clause (#47) excludes it, so
    # the LLM is never called for it again.
    assert stub_create.call_count == 1

    relations = await _fetch_relations(session_factory, anchor_id)
    assert len(relations) == 1
    assert relations[0].to_event_id == candidate_id


async def test_relate_stage_skips_event_malformed_on_both_attempts(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    # Structurally invalid JSON on both `call_structured` attempts ->
    # `json.JSONDecodeError` both times.
    stub_create.side_effect = [
        _FakeChatCompletion("not valid json"),
        _FakeChatCompletion("still not valid json"),
    ]

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Malformed Event")
        session.add(event)
        await session.commit()
        event_id = event.id

        result = await relate_stage(session, client=client)
        await session.commit()

    assert result == {"related": 0, "skipped": 1}
    assert stub_create.call_count == 2

    relations = await _fetch_relations(session_factory, event_id)
    assert relations == []


async def test_relate_stage_selection_requires_scored_and_unrelated(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion('{"relations": []}')

    async with session_factory() as session:
        watch = await _make_watch(session)
        not_yet_scored = _make_event(
            watch.id, embedding=ANCHOR_EMBEDDING, relevance=None, title="Not yet scored"
        )
        scored_unrelated = _make_event(
            watch.id, embedding=NEAR_EMBEDDING, relevance="low", title="Scored, unrelated"
        )
        session.add_all([not_yet_scored, scored_unrelated])
        await session.commit()
        not_yet_scored_id = not_yet_scored.id

        result = await relate_stage(session, client=client)
        await session.commit()

    # Only `scored_unrelated` is selected -> exactly one LLM call, an empty
    # `RelationSet` -> zero rows, counted toward neither "related" nor
    # "skipped".
    assert result == {"related": 0, "skipped": 0}
    assert stub_create.call_count == 1

    relations = await _fetch_relations(session_factory, not_yet_scored_id)
    assert relations == []


async def test_relate_stage_handles_empty_candidate_pool(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """`event` is the only row for its watch -- both candidate-finding
    functions return `[]`, the prompt renders the "(no candidate events
    found)" case, and a stubbed empty `RelationSet` is handled without
    error."""
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion('{"relations": []}')

    async with session_factory() as session:
        watch = await _make_watch(session)
        lonely = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Lonely Event")
        session.add(lonely)
        await session.commit()
        lonely_id = lonely.id

        result = await relate_stage(session, client=client)
        await session.commit()

    assert result == {"related": 0, "skipped": 0}
    assert stub_create.call_count == 1

    prompt_content = stub_create.call_args.kwargs["messages"][0]["content"]
    assert "(no candidate events found)" in prompt_content

    relations = await _fetch_relations(session_factory, lonely_id)
    assert relations == []


# ---------------------------------------------------------------------------
# related_at completion marker (#47)
# ---------------------------------------------------------------------------


async def test_relate_stage_sets_related_at_on_legitimately_empty_relation_set(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An event whose stubbed `RelationSet` is `{"relations": []}` -- the LLM
    legitimately proposing zero relations, not a validation failure -- gets
    `related_at` set from this single successful parse, and is not re-sent
    to the LLM on a second `relate_stage` call. This is the #47 gap itself:
    before this issue, such an event had no outbound `event_relation` row
    and was reselected forever."""
    client, stub_create = _client_with_stubbed_create()
    stub_create.return_value = _FakeChatCompletion('{"relations": []}')

    async with session_factory() as session:
        watch = await _make_watch(session)
        lonely = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Zero Relations")
        session.add(lonely)
        await session.commit()
        lonely_id = lonely.id

    async with session_factory() as session:
        first_result = await relate_stage(session, client=client)
        await session.commit()

    assert first_result == {"related": 0, "skipped": 0}
    assert stub_create.call_count == 1

    event_after_first_call = await _fetch_event(session_factory, lonely_id)
    assert event_after_first_call.related_at is not None

    async with session_factory() as session:
        second_result = await relate_stage(session, client=client)
        await session.commit()

    assert second_result == {"related": 0, "skipped": 0}
    # Not reselected -- the LLM is never called for it again.
    assert stub_create.call_count == 1

    relations = await _fetch_relations(session_factory, lonely_id)
    assert relations == []


async def test_relate_stage_sets_related_at_when_only_proposal_filtered_to_zero(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An event whose only proposed relation is dropped by the existing
    self-relation filter (so zero rows are persisted despite a non-empty
    `RelationSet`) also gets `related_at` set, and is not re-sent on a
    second call -- the filtered-to-zero case, distinct from the LLM
    legitimately returning `[]` outright."""
    client, stub_create = _client_with_stubbed_create()

    async with session_factory() as session:
        watch = await _make_watch(session)
        anchor = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Anchor")
        session.add(anchor)
        await session.commit()
        anchor_id = anchor.id

    # The only proposal is a self-relation -- filtered out, zero rows
    # persisted, despite a structurally non-empty `RelationSet`.
    stub_create.return_value = _FakeChatCompletion(
        f'{{"relations": [{{"event_id": "{anchor_id}", "relation": "similar", '
        f'"rationale": "Self-relation, should be dropped."}}]}}'
    )

    async with session_factory() as session:
        first_result = await relate_stage(session, client=client)
        await session.commit()

    assert first_result == {"related": 0, "skipped": 0}
    assert stub_create.call_count == 1

    event_after_first_call = await _fetch_event(session_factory, anchor_id)
    assert event_after_first_call.related_at is not None

    relations = await _fetch_relations(session_factory, anchor_id)
    assert relations == []

    async with session_factory() as session:
        second_result = await relate_stage(session, client=client)
        await session.commit()

    assert second_result == {"related": 0, "skipped": 0}
    # Not reselected -- the LLM is never called for it again, even though
    # zero relations were ever actually persisted for this event.
    assert stub_create.call_count == 1


async def test_relate_stage_keeps_related_at_null_and_resends_on_repeated_validation_failure(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An event that fails validation on both `call_structured` attempts
    keeps `related_at IS NULL` (same "leave the row unmodified, retry next
    run" precedent the stage already establishes for `skipped`) and *is*
    re-sent to the LLM on a second `relate_stage` call -- a transient LLM
    failure is still retried, unlike a genuine "nothing to relate" outcome."""
    client, stub_create = _client_with_stubbed_create()
    # Structurally invalid JSON on both `call_structured` attempts for the
    # first `relate_stage` call.
    stub_create.side_effect = [
        _FakeChatCompletion("not valid json"),
        _FakeChatCompletion("still not valid json"),
    ]

    async with session_factory() as session:
        watch = await _make_watch(session)
        event = _make_event(watch.id, embedding=ANCHOR_EMBEDDING, title="Always Malformed")
        session.add(event)
        await session.commit()
        event_id = event.id

    async with session_factory() as session:
        first_result = await relate_stage(session, client=client)
        await session.commit()

    assert first_result == {"related": 0, "skipped": 1}
    assert stub_create.call_count == 2

    event_after_first_call = await _fetch_event(session_factory, event_id)
    assert event_after_first_call.related_at is None

    # Second call: still selected (`related_at IS NULL`), and this time the
    # LLM succeeds with an empty `RelationSet`.
    stub_create.side_effect = None
    stub_create.return_value = _FakeChatCompletion('{"relations": []}')

    async with session_factory() as session:
        second_result = await relate_stage(session, client=client)
        await session.commit()

    assert second_result == {"related": 0, "skipped": 0}
    # Re-selected and re-sent: 2 attempts from the first call + 1 more here.
    assert stub_create.call_count == 3

    event_after_second_call = await _fetch_event(session_factory, event_id)
    assert event_after_second_call.related_at is not None
