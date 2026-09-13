"""Tests for src/nie/seed/run.py (issue #14).

Per _docs/testing-guidelines.md, DB-backed tests run against the real
Docker Compose Postgres, never mocked, and idempotency-focused acceptance
criteria are tested by calling the function under test twice and
asserting no duplication/clobbering -- not just that it runs once without
error.

Follows the `migrated_db`/`session_factory` fixture pattern from
`tests/test_db.py` and `tests/test_models.py`: a fresh engine per test
(rather than `nie.db`'s module-level singleton) so pooled asyncpg
connections stay bound to this test's own event loop.

The single test below calls `seed()` twice against the same
`session_factory`/engine, simulating a user edit (a `Watch.status`
change, a user-authored `ContextItem`, and a `NotificationPreference`
edit) in between, and asserts all four idempotency conditions from the
issue in one place -- they all hinge on the same seeded Silver watch, so
splitting them into separate tests would mean repeating the same
double-seed-plus-edit setup for each.

Unlike the rest of this test suite's `unique_slug`-per-test pattern
(`tests/test_models.py`), the Silver watch's slug is fixed
(`nie.seed.run.SILVER_WATCH_SLUG`) by design -- that's the whole point of
`seed()`'s upsert idempotency -- so the same `watch` row, and any
`kind="user"` `ContextItem` rows added against it, persist across
separate runs of this test suite against the same live database. The
`kind="user"` assertions below therefore compare counts/content
before-and-after the second `seed()` call within a single test run,
rather than asserting an absolute count of `1`, and give the row this
run adds a per-run-unique label so it can be found unambiguously.
"""

import re
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.models import Category, ContextItem, NotificationPreference, Watch
from nie.seed.run import CONTEXT_FILE, SILVER_WATCH_SLUG, seed

REPO_ROOT = Path(__file__).parent.parent

# Independently derived from the content file (not from run.py's parser),
# so this test would catch a heading count mismatch caused by a bug in
# either side.
HEADING_COUNT = len(re.findall(r"^## ", CONTEXT_FILE.read_text(), flags=re.MULTILINE))


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

    See `tests/test_models.py`'s identical fixture for why a fresh engine
    (rather than `nie.db`'s module-level singleton) is needed per test.
    """
    engine = create_engine()
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


async def test_seed_is_idempotent_and_preserves_edits_made_between_runs(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with session_factory() as session:
        await seed(session)

    async with session_factory() as session:
        watch_result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
        watch_id = watch_result.scalar_one().id

    # A per-run-unique label so this run's own `kind="user"` row can be
    # found unambiguously even if earlier runs of this test left rows of
    # their own against the same persistent "silver" watch.
    user_note_label = f"My note ({uuid.uuid4().hex[:8]})"

    # Simulate edits made between the two runs: a `Watch.status` change,
    # a user-authored `ContextItem` (kind="user"), and a
    # `NotificationPreference` edit -- like the future #36 preferences UI
    # would make. None of these should be reverted by the second `seed()`
    # call below.
    async with session_factory() as session:
        watch_result = await session.execute(select(Watch).where(Watch.id == watch_id))
        watch = watch_result.scalar_one()
        watch.status = "disabled"

        session.add(
            ContextItem(
                watch_id=watch_id,
                kind="user",
                label=user_note_label,
                body="Watch for ETF inflows.",
            )
        )

        preference_result = await session.execute(
            select(NotificationPreference).where(NotificationPreference.watch_id == watch_id)
        )
        preference = preference_result.scalar_one()
        preference.min_importance = "critical"
        preference.categories = ["market"]
        preference.channels = ["email"]

        await session.commit()

    async with session_factory() as session:
        pre_seed_user_result = await session.execute(
            select(ContextItem).where(
                ContextItem.watch_id == watch_id, ContextItem.kind == "user"
            )
        )
        user_count_before_second_seed = len(pre_seed_user_result.scalars().all())

    async with session_factory() as session:
        await seed(session)

    async with session_factory() as session:
        # Exactly one `watch` row with slug="silver", and its status
        # change from between the runs was not reverted by the upsert.
        watch_result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
        watches = watch_result.scalars().all()
        assert len(watches) == 1
        assert watches[0].id == watch_id
        assert watches[0].status == "disabled"

        # `context_item` `kind="system"` count matches the heading count,
        # and the `kind="user"` rows present before the second run --
        # including the one added above -- survived untouched (same count,
        # and this run's own row still present with its original content).
        system_result = await session.execute(
            select(ContextItem).where(
                ContextItem.watch_id == watch_id, ContextItem.kind == "system"
            )
        )
        system_items = system_result.scalars().all()
        assert len(system_items) == HEADING_COUNT

        user_result = await session.execute(
            select(ContextItem).where(
                ContextItem.watch_id == watch_id, ContextItem.kind == "user"
            )
        )
        user_items = user_result.scalars().all()
        assert len(user_items) == user_count_before_second_seed

        added_note = next(item for item in user_items if item.label == user_note_label)
        assert added_note.body == "Watch for ETF inflows."

        # Exactly 13 categories exist (seed_categories's own idempotency).
        category_result = await session.execute(select(Category))
        assert len(category_result.scalars().all()) == 13

        # Exactly one `notification_preference` row for the Silver watch,
        # and the edit made between runs was not reverted.
        preferences_result = await session.execute(
            select(NotificationPreference).where(NotificationPreference.watch_id == watch_id)
        )
        preferences = preferences_result.scalars().all()
        assert len(preferences) == 1
        assert preferences[0].min_importance == "critical"
        assert preferences[0].categories == ["market"]
        assert preferences[0].channels == ["email"]
