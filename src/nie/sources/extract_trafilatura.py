"""`TrafilaturaExtractor` -- fetches a URL and parses it with trafilatura.

Per `_docs/design.md`'s pipeline stage 2 ("fetch full text using
`TrafilaturaExtractor`"), the extractor owns the HTTP fetch itself: it
takes a bare URL, not already-fetched HTML. `trafilatura.fetch_url` is
called as a plain module-level reference (`trafilatura.fetch_url(...)`)
so tests can `monkeypatch.setattr` it to return fixture HTML instead of
making a live request. Both fetch failure and parse failure raise the
same `ExtractionError` -- see `nie.sources.base` for why.
"""

from __future__ import annotations

import trafilatura
from trafilatura.settings import Document

from nie.sources.base import ExtractedContent, ExtractionError


class TrafilaturaExtractor:
    """Downloads a URL and extracts its title and clean article text."""

    def extract(self, url: str) -> ExtractedContent:
        try:
            downloaded = trafilatura.fetch_url(url)
        except Exception as exc:
            raise ExtractionError(f"failed to fetch {url}") from exc

        if downloaded is None:
            raise ExtractionError(f"failed to fetch {url}")

        result = trafilatura.bare_extraction(downloaded, url=url, with_metadata=True)

        # `bare_extraction` returns `Document | dict | None`; `dict` only when
        # the deprecated `as_dict=True` is passed, which we never do -- the
        # isinstance check narrows the type for mypy rather than changing
        # behavior.
        if not isinstance(result, Document) or not result.text:
            raise ExtractionError(f"failed to parse {url}")

        return ExtractedContent(title=result.title or "", text=result.text)
