"""Triage stage (#24, `design.md` §5 stage 3).

`triage_stage` is a `StageFn` (per #18): for every `extracted` `source`
row it asks the LLM a cheap plausibility question -- "could this
plausibly be a meaningful Silver market event?" -- and drops obvious
noise (`status="triaged_out"` + `triage_note`) before any later stage
spends a real reasoning call on it. Replaces the
`STAGE_REGISTRY["triage"]` no-op from #18.

Unlike `match` (#22)/`adjudicate` (#25), triage *is* a `STAGE_REGISTRY`
stage: for every `extracted` source, independently, the answer is a
yes/no-ish verdict with a side effect on that row alone -- the same
"select a status batch, loop, mutate each row, return counts" shape as
`discover_stage`/`extract_stage`/`embed_stage`.

The prompt (`src/nie/llm/prompts/triage.md`) needs only the source's own
`title`/`content` plus fixed hardcoded Silver-watch framing text baked
into the prompt file itself -- no `context_item` query, no context
bundle. `design.md` §6's "context bundle" is reserved for the score
stage (#28), not triage.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.llm.client import LLMClient
from nie.models import Source

_PROMPT_PATH = Path(__file__).parent.parent / "llm" / "prompts" / "triage.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text()


class TriageResult(BaseModel):
    """The LLM's structured verdict for one `source` row, per `triage.md`."""

    plausible: bool
    note: str


def _build_messages(source: Source) -> list[dict[str, str]]:
    """Fill `triage.md` in with this row's `title`/`content` only."""
    prompt = _PROMPT_TEMPLATE.format(title=source.title, content=source.content)
    return [{"role": "user", "content": prompt}]


async def triage_stage(session: AsyncSession, *, client: LLMClient | None = None) -> dict[str, int]:
    """Ask the LLM a cheap plausibility question about every `extracted` row.

    `client` defaults to constructing its own `LLMClient()` when not
    given -- the optional param exists purely so tests can inject an
    already-stubbed client, the same way `tests/test_llm_client.py` stubs
    `LLMClient._client.chat.completions.create` on a constructed
    instance.

    Selects rows via `select(Source).where(Source.status == "extracted")`
    and processes them sequentially (no concurrency), same style as
    `extract_stage`. For each row, calls
    `client.call_structured(messages, TriageResult)` with `messages`
    built from `triage.md` filled in with that row's `title`/`content`
    only.

    - `plausible=False` -> `source.status = "triaged_out"` and
      `source.triage_note = result.note`.
    - `plausible=True` -> `status`/`triage_note` are left untouched (row
      stays `"extracted"`, ready for `embed_stage`).
    - If `call_structured` raises `json.JSONDecodeError` or
      `pydantic.ValidationError` for a row (still malformed after its own
      internal validate-then-retry-once), that row is left completely
      unmodified and the loop continues to the next row -- this failure
      must not abort the stage, matching `extract_stage`'s
      `ExtractionError` per-row isolation.
    - Any other exception (e.g. an `openai` SDK error meaning the API
      itself is unreachable/unauthorized) is **not** caught here -- it
      propagates out of `triage_stage` uncaught, same documented
      precedent as `extract_stage`. The runner's own try/except then
      records `stats["triage"] = {"error": ...}` for the run and
      continues to later stages.

    Does not call `session.commit()` -- only mutates already-tracked
    `Source` objects; #18's runner commits after the stage returns.

    Returns `{"triaged_out": N, "kept": M, "skipped": K}` -- negative
    verdicts, positive verdicts, and per-row parse/validation failures
    this call, respectively.
    """
    if client is None:
        client = LLMClient()

    result = await session.execute(select(Source).where(Source.status == "extracted"))
    sources = result.scalars().all()

    triaged_out_count = 0
    kept_count = 0
    skipped_count = 0
    for source in sources:
        messages = _build_messages(source)
        try:
            verdict = await client.call_structured(messages, TriageResult)
        except (json.JSONDecodeError, ValidationError):
            skipped_count += 1
            continue

        assert isinstance(verdict, TriageResult)
        if verdict.plausible:
            kept_count += 1
        else:
            source.status = "triaged_out"
            source.triage_note = verdict.note
            triaged_out_count += 1

    return {
        "triaged_out": triaged_out_count,
        "kept": kept_count,
        "skipped": skipped_count,
    }
