"""In-process scheduler wiring (issue #40, `design.md` §3/§18).

`design.md` §18's scheduler row names APScheduler's `AsyncIOScheduler`
explicitly ("In-process, no extra infra for MVP") -- not a bare
`asyncio.sleep` loop -- and §17 folds it into the single Uvicorn process
("Uvicorn (which also starts the APScheduler job)"). This module owns
both halves: the tick logic that actually runs the pipeline, and the
`AsyncIOScheduler` start/stop wiring that `src/nie/web/app.py`'s
`create_app()` lifespan calls.

`scheduler_tick` is deliberately a standalone, directly-callable async
function -- not only reachable by waiting out the real interval -- so
tests can call it directly without depending on real wall-clock
`POLL_INTERVAL_MINUTES` sleeps (per this issue's own test acceptance
criteria).
"""

from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.config import Settings
from nie.db import async_session_factory
from nie.models import Watch
from nie.pipeline.runner import run_pipeline
from nie.seed.run import SILVER_WATCH_SLUG

# Trigger value for a scheduler-initiated run. Must be the literal
# "schedule", not "scheduled" -- `PipelineRun.trigger` has a DB
# `CheckConstraint("trigger IN ('schedule', 'manual')")`
# (`alembic/versions/a0023c0d85d2_pipeline_run_table.py`); any other value
# fails the insert.
SCHEDULE_TRIGGER = "schedule"


async def scheduler_tick(
    session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
) -> None:
    """One scheduler interval firing, directly callable (not only via the
    real APScheduler interval) so tests never depend on real
    `POLL_INTERVAL_MINUTES` wall-clock sleeps.

    Looks up the Silver `Watch` row using the same query
    `_get_silver_watch` (`src/nie/web/routers/watch.py`) runs, but never
    raises when it's missing (a migrated-but-never-seeded DB) -- it skips
    silently instead, since an unhandled exception here would crash the
    scheduler thread/app process, not just one HTTP request.

    Calls `run_pipeline(session_factory=session_factory,
    trigger="schedule")` only when `watch.status == "enabled"`; a
    disabled watch is also a silent no-op.
    """
    async with session_factory() as session:
        result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
        watch = result.scalar_one_or_none()

    if watch is None or watch.status != "enabled":
        return

    await run_pipeline(session_factory=session_factory, trigger=SCHEDULE_TRIGGER)


def create_scheduler(settings: Settings | None = None) -> AsyncIOScheduler:
    """Build (but do not start) an `AsyncIOScheduler` with `scheduler_tick`
    registered on an interval trigger of `Settings().poll_interval_minutes`
    minutes.

    `max_instances` is left at APScheduler's default (1), so an
    in-progress `scheduler_tick` call blocks the next interval's job from
    starting a second, overlapping `run_pipeline` run.
    """
    settings = settings or Settings()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scheduler_tick,
        trigger=IntervalTrigger(minutes=settings.poll_interval_minutes),
        id="pipeline_scheduler_tick",
    )
    return scheduler
