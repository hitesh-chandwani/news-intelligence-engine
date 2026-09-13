"""Embed stage (#21, `design.md` §5 stage 4).

`embed_stage` is a `StageFn` (per #18): it fills in `embedding` for
`extracted` `source` rows that don't have one yet, using
`nie.embeddings.fastembed.embed_texts`, replacing the
`STAGE_REGISTRY["embed"]` no-op from #18.

Relies on the invariant that `status="extracted"` implies `content is
not None` (per `extract_stage`, #20) -- every selected row's `content`
is assumed usable for embedding without a `None` check.

Selection is `status == "extracted" AND embedding IS NULL`, so this
stage is idempotent: an `extracted` row that already has an `embedding`
is skipped on a later call, and every other status (`discovered`,
`extract_failed`, `triaged_out`, `processed`) is left untouched --
`ck_source_status` (#8) has no post-embed status value, so an embedded
row just gains an `embedding` and stays `status="extracted"`.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.embeddings.fastembed import embed_texts
from nie.models import Source


async def embed_stage(session: AsyncSession) -> dict[str, int]:
    """Embed `extracted` `source` rows with no `embedding` yet.

    Selects rows via
    `select(Source).where(Source.status == "extracted", Source.embedding.is_(None))`.
    When no rows match, returns `{"embedded": 0}` immediately without
    calling `embed_texts` at all -- no assumption that it accepts/no-ops
    cleanly on an empty batch.

    For each selected row, the text embedded is exactly
    `f"{source.title}\\n\\n{source.content}"` (title, blank line,
    content) -- documented here so later stages/tests can reproduce the
    same concatenation. Every selected row's text is embedded in a
    single `embed_texts(...)` batch call (not a per-row loop calling
    `embed_text`), and each result vector is assigned back to its row's
    `source.embedding` in the same order the rows were selected in.

    Does not change `source.status` and does not call
    `session.commit()` -- same as `discover_stage`/`extract_stage`;
    #18's runner commits after the stage returns.

    Returns `{"embedded": N}`, `N` = rows updated this call.
    """
    result = await session.execute(
        select(Source).where(Source.status == "extracted", Source.embedding.is_(None))
    )
    sources = result.scalars().all()

    if not sources:
        return {"embedded": 0}

    texts = [f"{source.title}\n\n{source.content}" for source in sources]
    vectors = embed_texts(texts)

    for source, vector in zip(sources, vectors, strict=True):
        source.embedding = vector

    return {"embedded": len(sources)}
