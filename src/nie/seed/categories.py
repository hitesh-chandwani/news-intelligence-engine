"""Idempotent seed function for the `category` table (issue #7).

Inserts the 13 initial event categories from `_docs/plan.md` §6. #14's seed
script imports and calls `seed_categories`; this module does not wire itself
into any CLI/entrypoint (out of scope for #7).
"""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import Category

# (slug, name) pairs, per `_docs/plan.md` §6 -- keyed by slug for the
# ON CONFLICT (slug) DO NOTHING upsert below.
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("supply", "Supply"),
    ("mining", "Mining"),
    ("demand", "Demand"),
    ("industrial", "Industrial"),
    ("market", "Market"),
    ("price", "Price"),
    ("inventory", "Inventory"),
    ("etf-investment", "ETF/Investment"),
    ("macro", "Macro"),
    ("geopolitical", "Geopolitical"),
    ("regulatory", "Regulatory"),
    ("company", "Company"),
    ("other", "Other"),
)


async def seed_categories(session: AsyncSession) -> None:
    """Insert the 13 initial categories, skipping any that already exist.

    Uses a Postgres ``INSERT ... ON CONFLICT (slug) DO NOTHING`` upsert so
    calling this repeatedly against the same database is a no-op after the
    first run -- no `IntegrityError` on re-seeding, no duplicate rows.
    """
    stmt = insert(Category).values(
        [{"slug": slug, "name": name} for slug, name in CATEGORIES]
    )
    stmt = stmt.on_conflict_do_nothing(index_elements=[Category.slug])
    await session.execute(stmt)
    await session.commit()
