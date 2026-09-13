"""Discovery provider abstraction.

`_docs/design.md` §7 is the authoritative spec for `CandidateItem` and
`DiscoveryProvider` -- §5's mention of the same fields is a summary, not a
second spec. This module is pure typing: no I/O, no DB session, no network.
`nie.models.Watch` is imported only as a type reference for `discover`'s
signature.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from nie.models import Watch


@dataclass(frozen=True, slots=True)
class CandidateItem:
    """One discovered candidate item.

    Field-for-field per `design.md` §7:
    `{url, title, source_name, published_at, snippet, content?, entities?}`.
    Only `content` and `entities` are optional -- `published_at` and
    `snippet` are required, unlike `nie.models.Source.published_at`
    (nullable there because extraction/backfill can leave it unknown).
    """

    url: str
    title: str
    source_name: str
    published_at: datetime
    snippet: str
    content: str | None = None
    entities: list[str] | None = None


class DiscoveryProvider(Protocol):
    """A source of `CandidateItem`s for a `Watch` (`design.md` §7).

    `discover` is **synchronous** -- unusual for this codebase (most I/O
    here is async), but exactly what `design.md` §7 specifies, so every
    provider (`StubProvider` now, `RssProvider` in #17) implements it as
    plain `def`, not `async def`. It returns an `Iterable`, not
    necessarily a `list` -- callers should not assume the result is sized
    or re-iterable.
    """

    name: str

    def discover(self, watch: Watch, since: datetime) -> Iterable[CandidateItem]: ...


@dataclass(frozen=True, slots=True)
class ExtractedContent:
    """Result of a successful `Extractor.extract` call.

    Both fields are required strings, never `None` -- if an underlying
    library returns no title for a page, an `Extractor` implementation
    (e.g. `TrafilaturaExtractor`) coerces it to `""`.
    """

    title: str
    text: str


class ExtractionError(Exception):
    """Raised by any `Extractor` on failure to produce `ExtractedContent`.

    Covers both fetch failure (the URL could not be downloaded) and parse
    failure (the downloaded HTML yielded no usable text) -- callers (the
    pipeline's extract stage, #20) only need to catch this one exception
    class to mark a `source` row `status='extract_failed'`.
    """


class Extractor(Protocol):
    """Fetches a URL and returns its clean article content.

    Unlike `DiscoveryProvider`, there is no `name` attribute: `design.md`
    has no `EXTRACTORS` config selecting among multiple implementations
    for MVP -- only `TrafilaturaExtractor` exists, so there's nothing to
    key on.
    """

    def extract(self, url: str) -> ExtractedContent: ...
