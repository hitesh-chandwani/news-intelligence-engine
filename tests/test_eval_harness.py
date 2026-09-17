"""Eval harness (#41, `design.md` §16).

A **manual**, real-LLM-call harness, distinct from `uv run pytest`: it
runs the real pipeline stages 3-8 (`triage`, `embed`, `match`,
`adjudicate`, `synthesize`, `score`; `design.md` §5) against the 30-50
hand-authored, hand-labelled fixtures under `tests/fixtures/eval/` and
reports how often the pipeline's real decisions agree with those labels.
It measures judgment *quality* on the configured free-tier model, so it
deliberately does **not** stub the LLM client the way every other
DB-backed pipeline test in this suite does -- stubbing would only prove
the stub returns what it was told, defeating the point of a quality eval
(see the issue's "Constraints" section).

Implemented as a pytest target (`design.md` §17's `eval` `uv run`/`make`
task, and §16's "a pytest target runs stages 3-8 against fixtures and
reports agreement"), gated behind the `eval` marker rather than a
`testpaths` exclusion: `pyproject.toml`'s `addopts = "-m 'not eval'"`
deselects this module from every default `uv run pytest` invocation (it
is still *collected* -- pytest still imports this file and discovers the
test function -- but the marker filter skips executing it, so no LLM call
is ever attempted by the default run; see that file's comment for the
full reasoning). Run it explicitly:

    uv run pytest tests/test_eval_harness.py -m eval -s

(`-s` shows the printed report; pytest captures stdout by default.)

## Why each fixture runs its own full stage cycle, not one bulk pass

`tests/fixtures/eval/*.json` fixtures are linked into chains (a
`new_event` fixture, plus 0+ `existing_event` fixtures that name it via
`matches_fixture`) so the harness can grade `adjudicate_stage`'s
new-vs-existing decision *and* which prior event it matched. For that
matching to have anything real to match against, the `new_event`
fixture's `Event` row must already exist in the database by the time its
`existing_event` follow-up is adjudicated. `adjudicate_stage` doesn't
create that row itself -- only `synthesize_stage` does, as a *separate*
`STAGE_REGISTRY` stage that runs after `adjudicate_stage` finishes its
whole batch -- so simply inserting every fixture as a `Source` row and
then calling `triage_stage`/`embed_stage`/`adjudicate_stage`/
`synthesize_stage`/`score_stage` once each (the way
`tests/test_pipeline_runner.py`'s end-to-end test chains stages 3-8
together) would adjudicate every fixture in the same batch, before any
of their events exist.

Instead, each fixture is inserted and driven through its own full
`triage -> embed -> adjudicate -> synthesize -> score` cycle, one fixture
at a time, in fixture-id order (chain heads are numbered before their
followups, e.g. `01-...` before `02-...`) -- so by the time a
`existing_event` fixture is processed, the `Event` it should match is
already a committed row a real `find_candidate_events` vector search can
find.

## Why a fresh, dedicated `Watch` per run

`find_candidate_events`/`find_nearest_events` (`match.py`, #22/#27) do
filter their candidate search by `Event.watch_id`, so running this
harness against the real seeded "Silver" watch would risk this run's
`new_event` fixtures being adjudicated against whatever real events
already exist there from actual pipeline runs against this same Compose
database -- contaminating the very agreement numbers this harness exists
to measure, and making the harness non-repeatable (a second run would see
the first run's fixture-events as real candidates too). Each run instead
creates its own uniquely-slugged `Watch` (`unique_slug`, same helper
`tests/test_pipeline_extract.py` defines) and seeds it with the same
Silver background context (`nie/seed/silver_context.md`) real production
scoring uses, so `score_stage`'s judgment quality is representative --
just scoped to this run's own fixture events, never anyone else's.

## Why stale global-selection rows are still cleared up front

`triage_stage`/`embed_stage`/`adjudicate_stage`/`synthesize_stage`/
`score_stage`'s own selection queries are all global (no `watch_id`
filter, by design -- see each stage's own module docstring), so a stray
row left mid-pipeline by an earlier test file sharing this never-
truncated Compose database (same "global-selection-query test-pollution"
precedent `tests/test_pipeline_triage.py`/`test_pipeline_adjudicate.py`/
`test_pipeline_synthesize.py`/`test_pipeline_score.py` each already work
around) would otherwise get swept into this harness's real-LLM stage
calls alongside our own fixtures -- consuming free-tier rate-limit budget
on data that isn't part of the labelled set, and, worse, showing up as a
false "candidate" or corrupting the shape-only agreement counts. The
autouse fixture below applies the same "repoint stray rows to a terminal
status/value" precedent those files use, once, before any fixture is
inserted.

## What this harness does NOT do yet

It only *reports* agreement -- it does not assert against a pass-bar
threshold, because none exists yet (that's this issue's own "record the
pass bar" acceptance criterion, and it can only be filled in from a real
run's actual numbers, never invented). Once a human runs this with a
real `LLM_API_KEY` and records the resulting percentages in `README.md`,
a follow-up can turn the relevant number(s) below into a real
`assert ... >= PASS_BAR`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.config import Settings
from nie.db import create_engine, create_session_factory
from nie.models import ContextItem, Event, EventSource, Source, Watch
from nie.pipeline.adjudicate import adjudicate_stage
from nie.pipeline.embed import embed_stage
from nie.pipeline.score import score_stage
from nie.pipeline.synthesize import synthesize_stage
from nie.pipeline.triage import triage_stage
from nie.seed.categories import seed_categories
from nie.seed.run import CONTEXT_FILE

pytestmark = pytest.mark.eval

REPO_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "eval"


def unique_slug(prefix: str) -> str:
    """Same helper `tests/test_pipeline_extract.py` defines -- a fresh,
    per-run-unique `Watch.slug` so this run's events never collide with
    (or get matched against) a prior run's."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def migrated_db() -> None:
    """Run `alembic upgrade head` against the live Compose DB. Same plain
    (sync) fixture as `tests/test_pipeline_runner.py`."""
    config = Config(str(REPO_ROOT / "alembic.ini"))
    upgrade(config, "head")


@pytest.fixture
async def session_factory(migrated_db: None) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory built from its own engine, scoped to this run."""
    engine = create_engine()
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
async def _clear_stale_pipeline_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Repoint every pre-existing row sitting in a global stage-selection
    out of that selection, before this run's own fixtures are inserted.
    See the module docstring's "Why stale global-selection rows are still
    cleared up front" section for the full reasoning; same precedent as
    `tests/test_pipeline_triage.py`/`test_pipeline_adjudicate.py`/
    `test_pipeline_synthesize.py`/`test_pipeline_score.py`'s own
    equivalent fixtures.
    """
    async with session_factory() as session:
        # Covers triage_stage's (`status == "extracted"`), embed_stage's
        # (`status == "extracted" AND embedding IS NULL`), and
        # adjudicate_stage's (`status == "extracted" AND embedding IS NOT
        # NULL`) selections in one update -- all three are subsets of
        # `status == "extracted"`.
        await session.execute(
            update(Source).where(Source.status == "extracted").values(status="processed")
        )
        # synthesize_stage's selection: `status == "adjudicated"`.
        await session.execute(
            update(Source).where(Source.status == "adjudicated").values(status="processed")
        )
        # score_stage's selection: `Event.relevance IS NULL`.
        await session.execute(
            update(Event).where(Event.relevance.is_(None)).values(relevance="irrelevant")
        )
        await session.commit()


@dataclass(frozen=True)
class EvalFixture:
    fixture_id: str
    url: str
    title: str
    source_name: str
    published_at: str | None
    content: str
    entities: list[str]
    expected_decision: str  # "new_event" | "existing_event"
    expected_match: str | None  # another fixture's id, only when "existing_event"
    expected_relevance: str
    expected_importance: str


def _load_fixtures() -> list[EvalFixture]:
    """Load every `tests/fixtures/eval/*.json` fixture, sorted by
    filename -- fixture ids are numbered so a chain's `new_event` head
    always sorts before its `existing_event` followups (see the module
    docstring)."""
    fixtures = []
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        expected = data["expected"]
        fixtures.append(
            EvalFixture(
                fixture_id=data["id"],
                url=data["url"],
                title=data["title"],
                source_name=data["source_name"],
                published_at=data.get("published_at"),
                content=data["content"],
                entities=data.get("entities", []),
                expected_decision=expected["decision"],
                expected_match=expected.get("matches_fixture"),
                expected_relevance=expected["relevance"],
                expected_importance=expected["importance"],
            )
        )
    return fixtures


async def _seed_eval_watch(session: AsyncSession) -> uuid.UUID:
    """Create this run's own dedicated `Watch`, with the real category
    table (shared/global, idempotent) and a system `ContextItem` carrying
    the same Silver background `score_stage` uses in production, so this
    run's judgment quality is representative of a real run."""
    watch = Watch(slug=unique_slug("eval-harness"), name="Eval Harness Watch", status="enabled")
    session.add(watch)
    await session.flush()

    await seed_categories(session)

    session.add(
        ContextItem(
            watch_id=watch.id,
            kind="system",
            label="Silver market overview (eval harness)",
            body=CONTEXT_FILE.read_text(),
        )
    )
    await session.commit()
    return watch.id


async def _run_fixture_through_pipeline(
    session: AsyncSession, watch_id: uuid.UUID, fixture: EvalFixture
) -> Source:
    """Insert `fixture` as a `Source` row and drive it through its own
    full `triage -> embed -> adjudicate -> synthesize -> score` cycle.

    Each stage call is global (per stage docstrings), but thanks to
    `_clear_stale_pipeline_rows` above and this function only ever being
    called once fully to completion per fixture before the next fixture
    is inserted, the only row any of these calls ever finds eligible is
    this fixture's own -- see the module docstring's "Why each fixture
    runs its own full stage cycle" section.
    """
    source = Source(
        watch_id=watch_id,
        url=fixture.url,
        title=fixture.title,
        source_name=fixture.source_name,
        published_at=(
            datetime.fromisoformat(fixture.published_at) if fixture.published_at else None
        ),
        content=fixture.content,
        entities=fixture.entities,
        status="extracted",
    )
    session.add(source)
    await session.commit()

    await triage_stage(session)
    await session.commit()
    await embed_stage(session)
    await session.commit()
    await adjudicate_stage(session)
    await session.commit()
    await synthesize_stage(session)
    await session.commit()
    await score_stage(session)
    await session.commit()

    await session.refresh(source)
    return source


def _classify_decision(source: Source) -> str:
    """Map a fixture's final `Source` row to one of the outcomes this
    harness grades against a fixture's `expected_decision`:

    - `"new_event"` / `"existing_event"` -- synthesize_stage completed,
      matching the two labels the fixtures use.
    - `"noise"` -- adjudicate_stage judged it noise (a real, legal
      outcome neither fixture label expects, so always a disagreement).
    - `"triaged_out"` -- triage_stage dropped it before adjudication.
    - `"unresolved"` -- a `call_structured` validation retry still failed
      at some stage (per-row skip), so no decision was ever reached.
    """
    if source.status == "triaged_out":
        return "triaged_out"
    if source.adjudication_decision == "noise":
        return "noise"
    if source.status == "processed" and source.adjudication_decision == "new":
        return "new_event"
    if source.status == "processed" and source.adjudication_decision == "existing":
        return "existing_event"
    return "unresolved"


async def _event_id_for_source(session: AsyncSession, source_id: uuid.UUID) -> uuid.UUID | None:
    result = await session.execute(
        select(EventSource.event_id).where(EventSource.source_id == source_id)
    )
    return result.scalars().first()


async def test_eval_harness_reports_agreement(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Run every `tests/fixtures/eval/*.json` fixture through the real
    pipeline stages 3-8 and print agreement against each fixture's
    hand-assigned label.

    Fails fast, before touching the database, with a clear and actionable
    message when no `LLM_API_KEY` is configured -- this harness makes
    real LLM calls (see the module docstring) and cannot run without one.
    """
    settings = Settings()
    if not settings.llm_api_key:
        pytest.fail(
            "LLM_API_KEY is not set -- the eval harness (#41) makes real calls "
            "to the configured LLM provider and cannot run without one. Set "
            "LLM_API_KEY (see .env.example) and re-run: "
            "uv run pytest tests/test_eval_harness.py -m eval -s",
            pytrace=False,
        )

    fixtures = _load_fixtures()
    assert len(fixtures) >= 30, (
        f"expected 30-50 eval fixtures under {FIXTURES_DIR}, found {len(fixtures)}"
    )

    async with session_factory() as session:
        watch_id = await _seed_eval_watch(session)

    # fixture_id -> the Event row it ended up linked to (or None if it
    # never reached synthesize), so later existing_event fixtures can
    # check they matched the *same* event their expected chain head did.
    event_id_by_fixture: dict[str, uuid.UUID | None] = {}
    decision_by_fixture: dict[str, str] = {}
    relevance_by_fixture: dict[str, str | None] = {}
    importance_by_fixture: dict[str, str | None] = {}

    decision_correct = 0
    match_evaluable = 0
    match_correct = 0
    relevance_evaluable = 0
    relevance_correct = 0
    importance_evaluable = 0
    importance_correct = 0

    for fixture in fixtures:
        async with session_factory() as session:
            source = await _run_fixture_through_pipeline(session, watch_id, fixture)
            actual_decision = _classify_decision(source)
            decision_by_fixture[fixture.fixture_id] = actual_decision

            event_id = await _event_id_for_source(session, source.id)
            event_id_by_fixture[fixture.fixture_id] = event_id

            relevance: str | None = None
            importance: str | None = None
            if event_id is not None:
                event = await session.get(Event, event_id)
                assert event is not None
                relevance = event.relevance
                importance = event.importance
            relevance_by_fixture[fixture.fixture_id] = relevance
            importance_by_fixture[fixture.fixture_id] = importance

        if actual_decision == fixture.expected_decision:
            decision_correct += 1

        if fixture.expected_decision == "existing_event" and fixture.expected_match is not None:
            expected_event_id = event_id_by_fixture.get(fixture.expected_match)
            if expected_event_id is not None:
                match_evaluable += 1
                if actual_decision == "existing_event" and event_id == expected_event_id:
                    match_correct += 1

        if relevance is not None:
            relevance_evaluable += 1
            if relevance == fixture.expected_relevance:
                relevance_correct += 1
        if importance is not None:
            importance_evaluable += 1
            if importance == fixture.expected_importance:
                importance_correct += 1

    def _pct(numerator: int, denominator: int) -> str:
        if denominator == 0:
            return "n/a (0 evaluable)"
        return f"{100 * numerator / denominator:.1f}% ({numerator}/{denominator})"

    report_lines = [
        "",
        "=== Eval harness report (#41) ===",
        f"Fixtures run: {len(fixtures)}",
        f"new_event/existing_event decision agreement: {_pct(decision_correct, len(fixtures))}",
        f"existing_event -> correct matched-event agreement: "
        f"{_pct(match_correct, match_evaluable)}",
        f"relevance exact-match agreement: {_pct(relevance_correct, relevance_evaluable)}",
        f"importance exact-match agreement: {_pct(importance_correct, importance_evaluable)}",
        "Per-fixture outcomes:",
    ]
    for fixture in fixtures:
        report_lines.append(
            f"  {fixture.fixture_id}: expected={fixture.expected_decision}"
            f"(match={fixture.expected_match}) actual={decision_by_fixture[fixture.fixture_id]} "
            f"relevance expected={fixture.expected_relevance} "
            f"actual={relevance_by_fixture[fixture.fixture_id]} "
            f"importance expected={fixture.expected_importance} "
            f"actual={importance_by_fixture[fixture.fixture_id]}"
        )
    report = "\n".join(report_lines)
    print(report)
