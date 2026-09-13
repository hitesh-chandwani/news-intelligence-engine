"""Discover stage (#19, `design.md` §5 stage 1).

`discover_stage` is a `StageFn` (per #18): it builds the active
`DiscoveryProvider`s from a fresh `Settings()`, calls each against the
single Silver watch, drops any candidate whose `url` already exists as a
`source.url` for that watch (or was already kept earlier in this same
call), and inserts the rest as `source` rows with `status="discovered"`.

This module owns the provider-construction wiring `DISCOVERY_PROVIDERS`/
`RSS_FEEDS` into real `StubProvider`/`RssProvider` instances -- neither
#17 nor #18 did this, and it's exactly what a working discover stage
needs to do (see the issue's "Resolving where provider-construction
wiring lives" note).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.config import Settings
from nie.models import Source, Watch
from nie.seed.run import SILVER_WATCH_SLUG
from nie.sources.base import DiscoveryProvider
from nie.sources.rss import RssProvider
from nie.sources.stub import StubProvider

# Repo-root-relative, not a config field: `design.md` §14's config table
# has no `STUB_FIXTURES_DIR` entry, matching `StubProvider`'s own
# docstring that whatever wires providers together points it at
# `tests/fixtures/sources/`. `discover.py` -> `pipeline` -> `nie` -> `src`
# -> repo root is 3 `parents` up.
STUB_FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sources"


def _build_providers(settings: Settings) -> list[DiscoveryProvider]:
    """Build one provider instance per name in `settings.discovery_providers`.

    An unrecognized name raises `ValueError` naming it rather than being
    silently skipped -- a typo'd or not-yet-implemented provider (e.g.
    `"websearch"`) would otherwise silently under-discover.
    """
    providers: list[DiscoveryProvider] = []
    for name in settings.discovery_providers:
        if name == "stub":
            providers.append(StubProvider(fixtures_dir=STUB_FIXTURES_DIR))
        elif name == "rss":
            providers.append(RssProvider(feed_urls=settings.rss_feeds))
        else:
            raise ValueError(f"Unrecognized discovery provider: {name!r}")
    return providers


async def discover_stage(session: AsyncSession) -> dict[str, int]:
    """Discover new `source` rows for the single Silver watch.

    Looks up the watch via `Watch.slug == SILVER_WATCH_SLUG`; no row found
    raises `LookupError`, which propagates -- #18's runner already catches
    any stage exception, rolls back, and records `{"error": str(exc)}"`.
    `Watch.status` is deliberately not read/branched on here: gating a
    disabled watch is #40's job (scheduler wiring), not this stage's.

    Dedup is a pre-query + filter, not insert-then-catch: existing
    `source.url` values for this watch are loaded into a `set[str]`
    first, then every candidate from every active provider (in
    `discovery_providers` order) is skipped if its `url` is already in
    that set, adding each newly-kept `url` as it's processed -- so a
    duplicate `url` appearing twice in one run (two fixtures, or the same
    URL from two providers) is caught too, not only duplicates already in
    the DB. This avoids poisoning the shared session/transaction with an
    `IntegrityError`, since there are no per-insert `SAVEPOINT`s here.

    Does not call `session.commit()` -- only `session.add()`s new `Source`
    rows; #18's runner commits after each stage returns.
    """
    settings = Settings()
    providers = _build_providers(settings)

    watch_result = await session.execute(select(Watch).where(Watch.slug == SILVER_WATCH_SLUG))
    watch = watch_result.scalar_one_or_none()
    if watch is None:
        raise LookupError(f"No watch found with slug {SILVER_WATCH_SLUG!r}")

    since = datetime.now(UTC) - timedelta(days=settings.dedup_window_days)

    existing_urls_result = await session.execute(
        select(Source.url).where(Source.watch_id == watch.id)
    )
    seen_urls: set[str] = set(existing_urls_result.scalars().all())

    discovered_count = 0
    for provider in providers:
        for candidate in provider.discover(watch, since):
            if candidate.url in seen_urls:
                continue
            seen_urls.add(candidate.url)
            session.add(
                Source(
                    watch_id=watch.id,
                    url=candidate.url,
                    title=candidate.title,
                    source_name=candidate.source_name,
                    published_at=candidate.published_at,
                    content=candidate.content,
                    entities=candidate.entities or [],
                    status="discovered",
                )
            )
            discovered_count += 1

    return {"discovered": discovered_count}
