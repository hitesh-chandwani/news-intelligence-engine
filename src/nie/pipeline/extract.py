"""Extract stage (#20, `design.md` §5 stage 2).

`extract_stage` is a `StageFn` (per #18): it fills in `content` for
`source` rows discovered without it, using `TrafilaturaExtractor` (#16),
replacing the `STAGE_REGISTRY["extract"]` no-op from #18.

Selection is on `status` alone (`"discovered"` or `"extract_failed"`),
not `status IN (...) AND content IS NULL` -- `design.md` §7 / §5 stage 1
allows a future `DiscoveryProvider` to supply `content` directly, and
#19 deliberately leaves such a row at `status="discovered"`, naming this
stage as the one that owns the transition off `"discovered"`. So each
selected row is branched on whether `content` is already present rather
than assumed to need extraction.

The "capped" half of `design.md` §5 stage 2's "retried next run, capped"
is out of scope here: the `source` table (#8) has no attempt-count
column, so this stage only implements uncapped "retried next run" --
`status="extract_failed"` rows are picked up again on the next call with
no limit. True capped retry is #45.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import Source
from nie.sources.base import ExtractionError
from nie.sources.extract_trafilatura import TrafilaturaExtractor


async def extract_stage(session: AsyncSession) -> dict[str, int]:
    """Extract full text for `discovered`/`extract_failed` `source` rows.

    Constructs one `TrafilaturaExtractor()` directly -- no `EXTRACTORS`
    config exists (`nie.sources.base.Extractor`'s own docstring notes
    there's nothing to key on for MVP).

    Selects rows via
    `select(Source).where(Source.status.in_(("discovered", "extract_failed")))`
    and processes them sequentially, in query order (no concurrent
    extraction -- matches `discover_stage`'s style):

    - A `"discovered"` row whose `content` is already non-`None`
      (provider-supplied) is transitioned straight to `"extracted"` with
      `extracted_at` set, without calling the extractor.
    - Otherwise (content is `None`, whether `"discovered"` or a retried
      `"extract_failed"`), the extractor is called as
      `await asyncio.to_thread(extractor.extract, source.url)` --
      `TrafilaturaExtractor.extract` is synchronous and does blocking
      network I/O, so running it via `to_thread` keeps this async stage
      from stalling the event loop.
        - Success: `content`/`extracted_at`/`status="extracted"` are set.
          `result.title` is deliberately not written back to
          `Source.title` -- `design.md` names only `content`/
          `extracted_at`/`status` for this stage.
        - `ExtractionError`: caught here so it can't fail the whole
          stage/run. `status="extract_failed"` is set; `content`/
          `extracted_at` are left `None`/unset. Any other exception type
          propagates uncaught, same as `discover_stage`'s `LookupError`.

    Does not call `session.commit()` -- only mutates already-tracked
    `Source` objects; #18's runner commits after the stage returns.

    Returns `{"extracted": N, "extract_failed": M}` -- `N` counts both
    the provider-supplied-content skip case and real extraction
    successes; `M` counts extraction failures this call.
    """
    extractor = TrafilaturaExtractor()

    result = await session.execute(
        select(Source).where(Source.status.in_(("discovered", "extract_failed")))
    )
    sources = result.scalars().all()

    extracted_count = 0
    extract_failed_count = 0
    for source in sources:
        if source.status == "discovered" and source.content is not None:
            source.extracted_at = datetime.now(UTC)
            source.status = "extracted"
            extracted_count += 1
            continue

        try:
            extracted = await asyncio.to_thread(extractor.extract, source.url)
        except ExtractionError:
            source.status = "extract_failed"
            extract_failed_count += 1
        else:
            source.content = extracted.text
            source.extracted_at = datetime.now(UTC)
            source.status = "extracted"
            extracted_count += 1

    return {"extracted": extracted_count, "extract_failed": extract_failed_count}
