"""The pipeline runner scaffolding (#18, `design.md` §5).

`run_pipeline` opens one `pipeline_run` row, calls an ordered list of
stage functions against one shared `AsyncSession`, records each stage's
returned counts into `pipeline_run.stats`, and closes the row `ok` /
`partial` / `failed` depending on whether any stage raised. Every
registered stage is currently a no-op placeholder -- `design.md` §5's 10
real stages land one-by-one starting with #19, each replacing its own
`STAGE_REGISTRY` entry without touching this loop.

This module deliberately knows nothing about pipeline business logic: a
stage receives only the run's shared session and looks up whatever rows
it needs itself (the single global "Silver" watch, per #14) -- it is not
handed a `Watch` or provider list by the runner.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import async_session_factory
from nie.models import PipelineRun
from nie.pipeline.discover import discover_stage
from nie.pipeline.embed import embed_stage
from nie.pipeline.extract import extract_stage

# Every stage is an async callable taking the run's single shared
# `AsyncSession` and returning its own `dict[str, int]` of counters (e.g.
# a future `discover` stage returns `{"discovered": 5}`).
StageFn = Callable[[AsyncSession], Awaitable[dict[str, int]]]


async def _not_yet_implemented(session: AsyncSession) -> dict[str, int]:
    """Shared no-op placeholder for every stage not yet implemented.

    Does no DB I/O -- #19 onward each replace their own `STAGE_REGISTRY`
    entry with a real implementation, one at a time, without touching
    `run_pipeline`'s loop.
    """
    return {}


# The 10 stages from `design.md` §5, in pipeline order. `discover` (#19),
# `extract` (#20), and `embed` (#21) are the first real stages; every
# other entry still maps to the shared no-op placeholder, swapped out
# one at a time in later issues.
STAGE_REGISTRY: list[tuple[str, StageFn]] = [
    ("discover", discover_stage),
    ("extract", extract_stage),
    ("triage", _not_yet_implemented),
    ("embed", embed_stage),
    ("adjudicate", _not_yet_implemented),
    ("synthesize", _not_yet_implemented),
    ("score", _not_yet_implemented),
    ("relate", _not_yet_implemented),
    ("notify", _not_yet_implemented),
]


async def run_pipeline(
    session_factory: async_sessionmaker[AsyncSession] = async_session_factory,
    trigger: str = "manual",
    stages: Sequence[tuple[str, StageFn]] = STAGE_REGISTRY,
) -> uuid.UUID:
    """Run every stage in `stages`, in order, against one shared session.

    Opens exactly one `AsyncSession` for the whole run (not one per
    stage) and inserts a `PipelineRun(trigger=trigger, status="running",
    stats={})` row, committing it before any stage runs -- an error here
    propagates rather than being swallowed, since there's no row yet to
    mark `failed`.

    Each stage is then invoked with that same session. On success, its
    returned dict is written to `stats[stage_name]` and committed before
    the next stage runs, so an earlier stage's committed writes survive
    even if a later stage fails. On failure, the exception is caught, the
    session is rolled back (clearing the aborted transaction so the next
    stage's queries don't fail on the same connection), and
    `stats[stage_name] = {"error": str(exception)}` is recorded instead --
    one stage's failure never stops the rest of the run.

    A rollback expires every object in the session (SQLAlchemy always
    expires on rollback, independent of `expire_on_commit`), and because
    this is an `AsyncSession`, an expired attribute can't be lazily
    reloaded by plain attribute access (that requires an awaited DB
    round trip) -- so the run row is explicitly `refresh()`-ed after a
    rollback before its `stats` are read and updated again.

    Once every stage has run, `finished_at` and a terminal status ("ok"
    if nothing raised, "failed" if every stage raised, "partial"
    otherwise) are set and committed.

    `stats` is reassigned (`{**old, stage_name: value}`) rather than
    mutated in place -- `PipelineRun.stats` is a plain `dict` column, not
    a SQLAlchemy `MutableDict`, so an in-place `stats[stage_name] = ...`
    would not be detected as a change and would be silently lost.
    """
    async with session_factory() as session:
        run = PipelineRun(trigger=trigger, status="running", stats={})
        session.add(run)
        await session.commit()
        run_id: uuid.UUID = run.id

        failure_count = 0
        for stage_name, stage_fn in stages:
            try:
                result = await stage_fn(session)
            except Exception as exc:
                await session.rollback()
                await session.refresh(run)
                failure_count += 1
                run.stats = {**run.stats, stage_name: {"error": str(exc)}}
            else:
                run.stats = {**run.stats, stage_name: result}
            await session.commit()

        total = len(stages)
        if failure_count == 0:
            status = "ok"
        elif failure_count == total:
            status = "failed"
        else:
            status = "partial"

        run.finished_at = datetime.now(UTC)
        run.status = status
        await session.commit()

    return run_id
