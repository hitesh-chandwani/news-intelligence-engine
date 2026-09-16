"""Tests for src/nie/web/app.py + src/nie/web/routers/watch.py (issue #34).

Per _docs/testing-guidelines.md, DB-backed tests run against the real
Docker Compose Postgres, never mocked. Follows the `migrated_db`/
`session_factory` fixture pattern from `tests/test_run.py`: a fresh engine
per test (rather than `nie.db`'s module-level singleton) so pooled asyncpg
connections stay bound to this test's own event loop.

The app under test is built fresh per test via `create_app()` (never the
module-level FastAPI singleton some frameworks use) with
`app.dependency_overrides[get_session]` pointed at the test's own
`session_factory`, and driven with `httpx.AsyncClient(transport=
ASGITransport(app=app))` -- no real server process, per issue #34's test
spec.

The 404 test never deletes the real Silver watch row -- this repo's test
DB is a shared, never-truncated Compose Postgres instance, and Postgres
enforces the FK constraint from `context_item.watch_id` (and others) at
`DELETE` time, not commit time, so an uncommitted-then-rolled-back delete
still fails before it gets anywhere near the rollback. Instead, the test
monkeypatches `nie.web.routers.watch.SILVER_WATCH_SLUG` (the module-level
constant `_get_silver_watch` queries by) to a slug that provably doesn't
exist in the DB, so the router's real, unmodified query against the real
`watch` table legitimately returns no row -- no row in the shared DB is
ever touched, deleted, or left in a bad state for other tests.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.seed.run import SILVER_WATCH_SLUG, seed
from nie.web.app import create_app
from nie.web.deps import get_session
from nie.web.routers import watch as watch_router

REPO_ROOT = Path(__file__).parent.parent

# Field names of `nie.schemas.PipelineRunSummary`, as serialized in
# `WatchStatusResponse.last_run` -- used by
# `test_enable_status_disable_status_transitions` to check that a non-None
# `last_run` is well-formed, without asserting on `pipeline_run` row
# history in the shared test DB.
_PIPELINE_RUN_SUMMARY_FIELDS = {
    "id",
    "trigger",
    "status",
    "started_at",
    "finished_at",
    "stats",
    "error",
}


def _is_pipeline_run_summary_shaped(value: object) -> bool:
    """`True` if `value` is a dict with exactly `PipelineRunSummary`'s keys."""
    return isinstance(value, dict) and value.keys() == _PIPELINE_RUN_SUMMARY_FIELDS


@pytest.fixture
def migrated_db() -> None:
    """Run `alembic upgrade head` against the live Compose DB.

    A plain (sync) fixture: Alembic drives its own event loop internally
    (see alembic/env.py), so this must run outside pytest-asyncio's loop
    for the test function. Re-running this is idempotent -- Alembic is a
    no-op when the database is already at the target revision.
    """
    config = Config(str(REPO_ROOT / "alembic.ini"))
    upgrade(config, "head")


@pytest.fixture
async def session_factory(migrated_db: None) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory built from its own engine, scoped to this test.

    See `tests/test_run.py`'s identical fixture for why a fresh engine
    (rather than `nie.db`'s module-level singleton) is needed per test.
    """
    engine = create_engine()
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.fixture
async def seeded_client(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncClient]:
    """A `create_app()` instance with `get_session` overridden to
    `session_factory`, seeded with the Silver watch, driven over
    `ASGITransport` (no real server process) via `httpx.AsyncClient`.
    """
    async with session_factory() as session:
        await seed(session)

    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def test_enable_status_disable_status_transitions(seeded_client: AsyncClient) -> None:
    """`POST /watch/enable` -> `GET /watch/status` -> `POST /watch/disable`
    -> `GET /watch/status`, asserting `status` transitions
    "enabled" -> "disabled" at each step.
    """
    enable_response = await seeded_client.post("/watch/enable")
    assert enable_response.status_code == 200
    assert enable_response.json()["status"] == "enabled"

    status_response = await seeded_client.get("/watch/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "enabled"
    assert status_response.json()["slug"] == SILVER_WATCH_SLUG
    # This repo's test DB is shared and never truncated, so other test
    # modules may have already created `pipeline_run` rows before this
    # test runs -- don't assume a pristine DB with zero rows. Either
    # `last_run` is `None` (no pipeline has ever run) or it's a
    # well-formed `PipelineRunSummary` dict; both are valid.
    last_run = status_response.json()["last_run"]
    assert last_run is None or _is_pipeline_run_summary_shaped(last_run)

    disable_response = await seeded_client.post("/watch/disable")
    assert disable_response.status_code == 200
    assert disable_response.json()["status"] == "disabled"

    status_response_2 = await seeded_client.get("/watch/status")
    assert status_response_2.status_code == 200
    assert status_response_2.json()["status"] == "disabled"


async def test_enable_and_disable_are_idempotent(seeded_client: AsyncClient) -> None:
    """Calling `/watch/enable` (or `/watch/disable`) when already in that
    state is a no-op `200`, not an error.
    """
    first = await seeded_client.post("/watch/enable")
    assert first.status_code == 200
    assert first.json()["status"] == "enabled"

    second = await seeded_client.post("/watch/enable")
    assert second.status_code == 200
    assert second.json()["status"] == "enabled"

    await seeded_client.post("/watch/disable")

    third = await seeded_client.post("/watch/disable")
    assert third.status_code == 200
    assert third.json()["status"] == "disabled"


@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("GET", "/watch/status"),
        ("POST", "/watch/enable"),
        ("POST", "/watch/disable"),
    ],
)
async def test_hx_request_header_gets_html_fragment(
    seeded_client: AsyncClient, method: str, url: str
) -> None:
    """An `HX-Request: true` request to any of the three endpoints gets
    back `text/html` containing the current status text, not JSON.
    """
    response = await seeded_client.request(method, url, headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Status" in response.text
    assert "enabled" in response.text or "disabled" in response.text


async def test_plain_request_gets_json(seeded_client: AsyncClient) -> None:
    """Without an `HX-Request` header, the response is JSON, not HTML."""
    response = await seeded_client.get("/watch/status")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("GET", "/watch/status"),
        ("POST", "/watch/enable"),
        ("POST", "/watch/disable"),
    ],
)
async def test_missing_silver_watch_returns_404(
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    url: str,
) -> None:
    """All three endpoints return `404` with a JSON `{"detail": ...}` body
    when the Silver watch row does not exist.

    This repo's test DB is a shared, never-truncated Compose Postgres
    instance -- actually deleting the real Silver watch row is not an
    option: Postgres enforces the FK constraint from
    `context_item.watch_id` (and others) at `DELETE` time, not commit
    time, so even an uncommitted, later-rolled-back delete raises a real
    `IntegrityError` before rollback ever comes into play.

    Instead, this monkeypatches `nie.web.routers.watch.SILVER_WATCH_SLUG`
    -- the module-level constant `_get_silver_watch` queries `Watch.slug`
    by -- to a slug that provably doesn't exist in the DB. The router's
    query then runs unmodified against the real `watch` table and
    legitimately finds no row, so no row anywhere in the shared DB is
    touched, deleted, or left in a bad state for other tests.
    """
    monkeypatch.setattr(watch_router, "SILVER_WATCH_SLUG", "nonexistent-watch-slug-for-404-test")

    async with session_factory() as session:
        await seed(session)

    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.request(method, url)

    assert response.status_code == 404
    assert "detail" in response.json()
