"""Tests for src/nie/pipeline/runner.py (issue #18).

DB-backed per `_docs/testing-guidelines.md`: runs against the live
Compose Postgres, never mocked. Follows `tests/test_run.py`'s
`migrated_db`/`session_factory` fixture pattern -- a fresh engine per
test (`nie.db.create_engine`/`create_session_factory`), not the
module-level singleton, so pooled asyncpg connections stay bound to this
test's own event loop.

`discover` (#19), `extract` (#20), `triage` (#24), and `embed` (#21) are
the first `STAGE_REGISTRY` entries to become real stages rather than the
shared no-op placeholder -- this module's own docstring anticipated
exactly this ("each replacing its own `STAGE_REGISTRY` entry").
`test_all_stages_running_end_to_end_produce_an_ok_run` below isolates
env and seeds the Silver watch so the real `discover_stage` succeeds
without a network call, and only asserts that it *succeeded* (a
`"discovered"` key), leaving the exact insert count to
`tests/test_pipeline_discover.py` -- this file's own concern is the
runner loop, not discover's business logic, and a previous run of the
suite may have already discovered these same fixtures for this
persistent watch. `trafilatura.fetch_url` is monkeypatched to fail every
fetch (same pattern as `tests/test_extract_trafilatura.py`) so that
`extract_stage`, now real too, never makes a live network call against
whatever the real `discover_stage` just inserted -- this test only
asserts `extract`'s stats have the expected shape (per #20's stage
contract), same as `discover`'s, not exact counts. `embed_stage`, now
real too (#21), gets the same shape-only treatment: its selection has no
`watch_id` filter either, so rows left behind by other test files
sharing this never-truncated DB can make its exact count non-
deterministic here.

`triage_stage` (#24) is now real too, and unlike `discover`/`extract`/
`embed` it needs an `LLMClient` -- `run_pipeline` calls every
`STAGE_REGISTRY` entry as `stage_fn(session)` (no way to pass a
per-stage `client` through the registry), so this test monkeypatches
`nie.pipeline.triage.LLMClient` itself (the name that module imports
and constructs when `triage_stage`'s own `client` param is left `None`,
exactly as it is via the registry) to a stub whose `call_structured`
always returns a positive verdict, with no real network call/API key --
per `_docs/testing-guidelines.md`. Same shape-only treatment as
`embed`'s: `triage_stage`'s selection has no `watch_id` filter either, so
its exact counts aren't asserted here.

`adjudicate_stage` (#25) is now real too, same `LLMClient`-stubbing
treatment as `triage_stage`'s: `nie.pipeline.adjudicate.LLMClient` is
monkeypatched to a stub whose `call_structured` always returns a
`"noise"` verdict (the simplest legal `AdjudicationResult` -- no
candidate `event_id` membership to satisfy), so this run makes no real
LLM call and needs no API key. Same shape-only treatment as `triage`'s:
`adjudicate_stage`'s selection has no `watch_id` filter either, so its
exact counts aren't asserted here.

`synthesize_stage` (#26) is now real too, same `LLMClient`-stubbing
treatment as `triage_stage`'s/`adjudicate_stage`'s:
`nie.pipeline.synthesize.LLMClient` is monkeypatched to a stub whose
`call_structured` always returns a legal `EventRecord` (`categories =
["other"]`, guaranteed to exist since #14's `seed()` call below seeds
the category table too), so this run makes no real LLM call and needs
no API key. `synthesize_stage`'s own selection (`status ==
"adjudicated"`) has no `watch_id` filter either, so -- like
`triage`/`adjudicate` -- this test doesn't assert its exact counts, only
that its stats have the expected shape; `tests/
test_pipeline_synthesize.py` owns synthesize's actual business-logic
coverage. Note this test file isn't in issue #26's own "Files:"
constraint list, but wiring the real stage into `STAGE_REGISTRY` (the
issue's own acceptance criteria) breaks this pre-existing end-to-end
assertion otherwise, exactly as #24/#25 already needed the same
treatment here -- same precedent, not a new one.

`score_stage` (#28) is now real too, same `LLMClient`-stubbing treatment
as `synthesize_stage`'s: `nie.pipeline.score.LLMClient` is monkeypatched
to a stub whose `call_structured` always returns a legal `ScoreResult`
(`relevance="irrelevant"`, the simplest legal verdict -- still fills in
every other field, per #28's uniform-write acceptance criterion), so this
run makes no real LLM call and needs no API key. `score_stage`'s own
selection (`Event.relevance.is_(None)`) has no `watch_id` filter either,
so -- like `synthesize`'s -- this test doesn't assert its exact counts,
only that its stats have the expected shape; `tests/test_pipeline_score.py`
owns score's actual business-logic coverage. Same "not in the issue's own
Files: list, but wiring into STAGE_REGISTRY breaks this pre-existing
end-to-end assertion otherwise" precedent as #24/#25/#26 above.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import PipelineRun
from nie.pipeline import adjudicate as adjudicate_module
from nie.pipeline import score as score_module
from nie.pipeline import synthesize as synthesize_module
from nie.pipeline import triage as triage_module
from nie.pipeline.adjudicate import AdjudicationResult
from nie.pipeline.runner import STAGE_REGISTRY, run_pipeline
from nie.pipeline.score import ScoreResult
from nie.pipeline.synthesize import EventRecord
from nie.pipeline.triage import TriageResult
from nie.seed.run import seed
from nie.sources import extract_trafilatura

REPO_ROOT = Path(__file__).parent.parent


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


async def _succeeding_stage(session: AsyncSession) -> dict[str, int]:
    return {"count": 3}


async def _raising_stage(session: AsyncSession) -> dict[str, int]:
    raise ValueError("boom")


class _StubTriageClient:
    """No-network stand-in for `LLMClient` -- always a positive verdict.

    `triage_stage` only ever calls `call_structured`, so this stub
    implements just that one method rather than the full `LLMClient`
    surface.
    """

    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[TriageResult]
    ) -> TriageResult:
        return response_model(plausible=True, note="stub triage verdict")


class _StubAdjudicateClient:
    """No-network stand-in for `LLMClient` -- always a `"noise"` verdict.

    `adjudicate_stage` only ever calls `call_structured`, so this stub
    implements just that one method. `"noise"` is the simplest legal
    `AdjudicationResult` -- unlike `"existing"`, it needs no candidate
    `event_id` to satisfy the membership check `adjudicate_stage` runs
    itself.
    """

    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[AdjudicationResult]
    ) -> AdjudicationResult:
        return response_model(decision="noise", event_id=None, materiality="none")


class _StubSynthesizeClient:
    """No-network stand-in for `LLMClient` -- always a legal `EventRecord`.

    `synthesize_stage` only ever calls `call_structured`, so this stub
    implements just that one method. `categories = ["other"]` is
    guaranteed to validate against the seeded category table (#14's
    `seed()` call below seeds it).
    """

    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[EventRecord]
    ) -> EventRecord:
        return response_model(
            title="Stub Event",
            fact_summary="Stub fact summary.",
            interpretation="Stub interpretation.",
            event_date=None,
            entities=[],
            categories=["other"],
        )


class _StubScoreClient:
    """No-network stand-in for `LLMClient` -- always an `"irrelevant"`
    verdict.

    `score_stage` only ever calls `call_structured`, so this stub
    implements just that one method. `"irrelevant"` is the simplest legal
    `ScoreResult` -- still fills in every other field, per #28's
    uniform-write acceptance criterion (no special-casing `irrelevant`).
    """

    async def call_structured(
        self, messages: list[dict[str, str]], response_model: type[ScoreResult]
    ) -> ScoreResult:
        return response_model(
            relevance="irrelevant",
            importance="low",
            impact_direction="neutral",
            impact_reason="Stub score verdict.",
            impact_confidence="low",
        )


async def test_all_stages_running_end_to_end_produce_an_ok_run(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # discover_stage (#19) builds its own Settings() -- force stub-only
    # discovery so this run makes no network call and doesn't depend on
    # RssProvider, and seed the Silver watch discover_stage looks up.
    monkeypatch.setenv("DISCOVERY_PROVIDERS", "stub")
    # extract_stage (#20) is now real too: whatever discover_stage just
    # inserted for the Silver watch gets picked up in the same run, so
    # trafilatura.fetch_url is monkeypatched to fail every fetch -- this
    # test only cares about the runner loop, not extraction outcomes, and
    # a failed extraction (status="extract_failed") is still a fully
    # valid, non-erroring stage result.
    monkeypatch.setattr(extract_trafilatura.trafilatura, "fetch_url", lambda url: None)
    # triage_stage (#24) is now real too -- stub the `LLMClient` it
    # constructs for itself (no `client` is passed through the registry)
    # so this run makes no real LLM call and needs no API key.
    monkeypatch.setattr(triage_module, "LLMClient", lambda *args, **kwargs: _StubTriageClient())
    # adjudicate_stage (#25) is now real too -- stub the `LLMClient` it
    # constructs for itself (no `client` is passed through the registry)
    # so this run makes no real LLM call and needs no API key.
    monkeypatch.setattr(
        adjudicate_module, "LLMClient", lambda *args, **kwargs: _StubAdjudicateClient()
    )
    # synthesize_stage (#26) is now real too -- stub the `LLMClient` it
    # constructs for itself (no `client` is passed through the registry)
    # so this run makes no real LLM call and needs no API key.
    monkeypatch.setattr(
        synthesize_module, "LLMClient", lambda *args, **kwargs: _StubSynthesizeClient()
    )
    # score_stage (#28) is now real too -- stub the `LLMClient` it
    # constructs for itself (no `client` is passed through the registry)
    # so this run makes no real LLM call and needs no API key.
    monkeypatch.setattr(score_module, "LLMClient", lambda *args, **kwargs: _StubScoreClient())
    async with session_factory() as session:
        await seed(session)

    run_id = await run_pipeline(session_factory, trigger="manual")

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.trigger == "manual"
        assert run.status == "ok"
        assert run.finished_at is not None
        assert run.finished_at >= run.started_at
        assert set(run.stats["discover"]) == {"discovered"}
        assert isinstance(run.stats["discover"]["discovered"], int)
        assert set(run.stats["extract"]) == {"extracted", "extract_failed"}
        assert isinstance(run.stats["extract"]["extracted"], int)
        assert isinstance(run.stats["extract"]["extract_failed"], int)
        # embed_stage (#21) is now real too, same as discover/extract: its
        # global `status == "extracted" AND embedding IS NULL` selection
        # (no watch_id filter, by design) can pick up rows left behind by
        # other test files sharing this never-truncated DB, so only the
        # shape of its stats is asserted here, not an exact count.
        assert set(run.stats["embed"]) == {"embedded"}
        assert isinstance(run.stats["embed"]["embedded"], int)
        # triage_stage (#24) is now real too, same shape-only treatment as
        # embed's above: its global `status == "extracted"` selection (no
        # `watch_id` filter, by design) can pick up rows left behind by
        # other test files sharing this never-truncated DB.
        assert set(run.stats["triage"]) == {"triaged_out", "kept", "skipped"}
        assert isinstance(run.stats["triage"]["triaged_out"], int)
        assert isinstance(run.stats["triage"]["kept"], int)
        assert isinstance(run.stats["triage"]["skipped"], int)
        # adjudicate_stage (#25) is now real too, same shape-only treatment
        # as triage's above: its global `status == "extracted" AND
        # embedding IS NOT NULL` selection (no `watch_id` filter, by
        # design) can pick up rows left behind by other test files sharing
        # this never-truncated DB.
        assert set(run.stats["adjudicate"]) == {"new", "existing", "noise", "skipped"}
        assert isinstance(run.stats["adjudicate"]["new"], int)
        assert isinstance(run.stats["adjudicate"]["existing"], int)
        assert isinstance(run.stats["adjudicate"]["noise"], int)
        assert isinstance(run.stats["adjudicate"]["skipped"], int)
        # synthesize_stage (#26) is now real too, same shape-only treatment
        # as adjudicate's above: its global `status == "adjudicated"`
        # selection (no `watch_id` filter, by design) can pick up rows
        # left behind by other test files sharing this never-truncated DB.
        assert set(run.stats["synthesize"]) == {
            "new",
            "existing_updated",
            "existing_linked",
            "skipped",
        }
        assert isinstance(run.stats["synthesize"]["new"], int)
        assert isinstance(run.stats["synthesize"]["existing_updated"], int)
        assert isinstance(run.stats["synthesize"]["existing_linked"], int)
        assert isinstance(run.stats["synthesize"]["skipped"], int)
        # score_stage (#28) is now real too, same shape-only treatment as
        # synthesize's above: its global `Event.relevance.is_(None)`
        # selection (no `watch_id` filter, by design) can pick up rows
        # left behind by other test files sharing this never-truncated DB.
        assert set(run.stats["score"]) == {"irrelevant", "scored", "skipped"}
        assert isinstance(run.stats["score"]["irrelevant"], int)
        assert isinstance(run.stats["score"]["scored"], int)
        assert isinstance(run.stats["score"]["skipped"], int)
        for name, _ in STAGE_REGISTRY:
            if name not in (
                "discover",
                "extract",
                "triage",
                "embed",
                "adjudicate",
                "synthesize",
                "score",
            ):
                assert run.stats[name] == {}


async def test_mixed_stages_produce_a_partial_run_and_keep_running_after_a_failure(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ran_after_failure = False

    async def _stage_after_failure(session: AsyncSession) -> dict[str, int]:
        nonlocal ran_after_failure
        ran_after_failure = True
        return {"after": 1}

    stages = [
        ("succeeds", _succeeding_stage),
        ("fails", _raising_stage),
        ("after", _stage_after_failure),
    ]

    run_id = await run_pipeline(session_factory, trigger="manual", stages=stages)

    assert ran_after_failure is True

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == "partial"
        assert run.stats["succeeds"] == {"count": 3}
        assert run.stats["fails"] == {"error": "boom"}
        assert run.stats["after"] == {"after": 1}


async def test_all_raising_stages_produce_a_failed_run(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    stages = [
        ("first", _raising_stage),
        ("second", _raising_stage),
    ]

    run_id = await run_pipeline(session_factory, trigger="manual", stages=stages)

    async with session_factory() as session:
        run = await session.get(PipelineRun, run_id)
        assert run is not None
        assert run.status == "failed"
        assert run.stats["first"] == {"error": "boom"}
        assert run.stats["second"] == {"error": "boom"}
