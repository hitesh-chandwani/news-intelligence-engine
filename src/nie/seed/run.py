"""Idempotent seed script for the Silver watch (issue #14).

Running ``uv run python -m nie.seed.run`` against a freshly migrated,
empty database leaves it with a usable Silver watch: the ``watch`` row,
its system background context loaded from ``silver_context.md``, the 13
event categories (#7, via `nie.seed.categories.seed_categories`), and a
default ``notification_preference`` row. Running it again leaves the
database in exactly the same state -- no duplicates, no clobbering of
anything a user has since edited.

Each of the four steps below uses the idempotency strategy the natural
key of its table allows:

- ``Watch``/``NotificationPreference`` have a natural unique key (``slug``,
  ``watch_id`` respectively), so they use the same
  ``INSERT ... ON CONFLICT (...) DO NOTHING`` idiom
  `nie.seed.categories.seed_categories` already uses for `Category.slug`.
- ``Category`` seeding is entirely delegated to `seed_categories` -- this
  module does not re-implement or duplicate that logic.
- ``ContextItem`` has no natural unique key to upsert on (`design.md` §4
  gives it only a synthetic ``id``), so it uses a delete-then-reinsert
  (replace) strategy instead, scoped to `watch_id` + `kind="system"` only.
  This never touches `kind="user"` rows, which are out of scope here (#35).
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from nie.db import async_session_factory
from nie.models import ContextItem, NotificationPreference, Watch
from nie.seed.categories import seed_categories

SILVER_WATCH_SLUG = "silver"
SILVER_WATCH_NAME = "Silver Commodity"

CONTEXT_FILE = Path(__file__).parent / "silver_context.md"

# Matches a `##` heading line and captures its text, so `_parse_context_items`
# can split the file into (label, body) pairs without a Markdown library.
_HEADING_RE = re.compile(r"^## (?P<label>.+)$", re.MULTILINE)


def _parse_context_items(markdown: str) -> list[tuple[str, str]]:
    """Split ``markdown`` into ``(label, body)`` pairs, one per `##` heading.

    ``label`` is the heading text; ``body`` is everything between that
    heading and the next (or end of file), stripped of leading/trailing
    whitespace.
    """
    headings = list(_HEADING_RE.finditer(markdown))
    items: list[tuple[str, str]] = []
    for i, match in enumerate(headings):
        label = match.group("label").strip()
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        body = markdown[start:end].strip()
        items.append((label, body))
    return items


async def _seed_watch(session: AsyncSession) -> uuid.UUID:
    """Upsert the Silver `Watch` row and return its id.

    Uses `INSERT ... ON CONFLICT (slug) DO NOTHING`, the same idiom
    `seed_categories` uses for `Category.slug` -- then looks the row up by
    `slug` to get its `id`, needed whether this run inserted it or it
    already existed.
    """
    stmt = insert(Watch).values(
        slug=SILVER_WATCH_SLUG, name=SILVER_WATCH_NAME, status="enabled"
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=[Watch.slug])
    await session.execute(stmt)

    result = await session.execute(select(Watch.id).where(Watch.slug == SILVER_WATCH_SLUG))
    return result.scalar_one()


async def _seed_context_items(session: AsyncSession, watch_id: uuid.UUID) -> None:
    """Replace the Silver watch's `kind="system"` `ContextItem` rows.

    Deletes any existing `kind="system"` rows for this watch, then inserts
    the freshly parsed rows from `silver_context.md`. `kind="user"` rows are
    never touched -- the delete is scoped to `watch_id` + `kind="system"`.
    """
    items = _parse_context_items(CONTEXT_FILE.read_text())

    await session.execute(
        delete(ContextItem).where(
            ContextItem.watch_id == watch_id, ContextItem.kind == "system"
        )
    )
    session.add_all(
        [
            ContextItem(watch_id=watch_id, kind="system", label=label, body=body)
            for label, body in items
        ]
    )


async def _seed_notification_preference(session: AsyncSession, watch_id: uuid.UUID) -> None:
    """Upsert the default `NotificationPreference` row for the Silver watch.

    Uses `INSERT ... ON CONFLICT (watch_id) DO NOTHING` -- `watch_id` is
    this table's primary key, so this is a natural-key upsert like
    `Watch`'s. Never overwrites an existing row, so a user edit made via
    the future #36 preferences UI survives a re-run.
    """
    stmt = insert(NotificationPreference).values(
        watch_id=watch_id, min_importance="medium", categories=[], channels=[]
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=[NotificationPreference.watch_id])
    await session.execute(stmt)


async def seed(session: AsyncSession) -> None:
    """Seed the Silver watch, its system context, categories, and default
    notification preference. Safe to call repeatedly against the same
    database (see module docstring for the per-table idempotency strategy).
    """
    watch_id = await _seed_watch(session)
    await seed_categories(session)
    await _seed_context_items(session, watch_id)
    await _seed_notification_preference(session, watch_id)
    await session.commit()


async def _main() -> None:
    async with async_session_factory() as session:
        await seed(session)


if __name__ == "__main__":
    asyncio.run(_main())
