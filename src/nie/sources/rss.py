"""`RssProvider` -- parses a fixed list of RSS/Atom feed URLs via `feedparser`.

Per `_docs/design.md` §7: discovers real Silver-related articles both from
curated single-publisher feeds (e.g. Kitco, Mining.com, Reuters) and from a
Google News RSS query feed. Google News entries link to
`news.google.com/rss/articles/...` redirect pages rather than the publisher
URL; this module resolves that one, narrowly-scoped redirect shape to the
real publisher URL before returning the `CandidateItem`, so #16's
`TrafilaturaExtractor` (which does *not* do this resolution) receives a
fetchable article URL.

This module is pure and independent of the DB/pipeline: it does not select
providers from `DISCOVERY_PROVIDERS`, read `RSS_FEEDS` config, dedupe
against `source.url`, or write any row -- that wiring is #18/#19.
"""

from __future__ import annotations

import calendar
import logging
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from html import unescape
from typing import Any
from urllib.parse import urlparse

# Re-exported (`as urlopen`) so mypy's `--strict --no-implicit-reexport`
# treats `rss.urlopen` as part of this module's public interface, and
# referenced below as a bare module-level call -- same convention #16
# established for `trafilatura.fetch_url` -- so a test can
# `monkeypatch.setattr(rss, "urlopen", ...)`.
from urllib.request import urlopen as urlopen

import feedparser

from nie.models import Watch
from nie.sources.base import CandidateItem

logger = logging.getLogger(__name__)

# Some publishers (e.g. mining.com) return 403 to feedparser's default
# user agent while serving the same feed fine to a browser -- matches
# `extract_trafilatura.py`'s `_BROWSER_USER_AGENT` fix for the same class
# of issue on the article-fetch side.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")

_GOOGLE_NEWS_HOST = "news.google.com"
_GOOGLE_NEWS_PATH_PREFIX = "/rss/articles/"


class RssProvider:
    """Parses a fixed list of RSS/Atom feed URLs into `CandidateItem`s.

    `feed_urls` is a required constructor argument with no default -- each
    entry is anything `feedparser.parse` accepts (a live URL in production,
    a local file path in tests) -- mirroring `StubProvider`'s required
    `fixtures_dir` from #15. Wiring config's `RSS_FEEDS` env var into this
    constructor, and registering `"rss"` under `DISCOVERY_PROVIDERS`, is
    #18, not this module's job.
    """

    name = "rss"

    def __init__(self, feed_urls: list[str]) -> None:
        self.feed_urls = feed_urls

    def discover(self, watch: Watch, since: datetime) -> Iterable[CandidateItem]:
        """Yield one `CandidateItem` per feed entry across all `feed_urls`.

        Both `watch` and `since` are ignored: every call parses the same
        `feed_urls` list, in list order, regardless of which watch or
        `since` timestamp is passed. Filtering by `since` / deduping
        against the DB is #19's job (the discover *stage*, not this
        provider).
        """
        for url in self.feed_urls:
            yield from _parse_feed(url)


def _parse_feed(url: str) -> Iterable[CandidateItem]:
    """Parse one feed URL, yielding its entries as `CandidateItem`s.

    A feed that raises, or that comes back `bozo` with no entries, is
    logged and skipped -- one bad/unreachable feed in `feed_urls` must not
    break discovery for every other feed in the list.
    """
    try:
        parsed = feedparser.parse(url, agent=_BROWSER_USER_AGENT)
    except Exception:
        logger.warning("rss: failed to parse feed %s", url, exc_info=True)
        return

    if parsed.get("bozo") and not parsed.entries:
        logger.warning(
            "rss: feed %s returned no entries (bozo_exception=%r)",
            url,
            parsed.get("bozo_exception"),
        )
        return

    feed_title: str = parsed.feed.get("title", "") if "feed" in parsed else ""

    for entry in parsed.entries:
        item = _parse_entry(entry, feed_title, url)
        if item is not None:
            yield item


def _parse_entry(entry: Any, feed_title: str, feed_url: str) -> CandidateItem | None:
    link = entry.get("link")
    title = entry.get("title")
    published_parsed = entry.get("published_parsed")

    if not link or not title or not published_parsed:
        logger.warning(
            "rss: skipping entry from %s missing required field(s) "
            "(link=%r, title=%r, published_parsed=%r)",
            feed_url,
            link,
            title,
            published_parsed,
        )
        return None

    source = entry.get("source")
    source_name: str = source.get("title", feed_title) if source else feed_title

    published_at = datetime.fromtimestamp(calendar.timegm(published_parsed), tz=UTC)

    snippet = _strip_html(entry.get("summary", ""))

    return CandidateItem(
        url=_resolve_url(link),
        title=title,
        source_name=source_name,
        published_at=published_at,
        snippet=snippet,
        content=None,
        entities=None,
    )


def _strip_html(text: str) -> str:
    """Strip HTML tags from `text`, unescaping entities, e.g. `&amp;`."""
    return unescape(_TAG_RE.sub("", text)).strip()


def _is_google_news_redirect(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == _GOOGLE_NEWS_HOST and parsed.path.startswith(
        _GOOGLE_NEWS_PATH_PREFIX
    )


def _resolve_url(url: str) -> str:
    """Resolve a Google News redirect link to its publisher URL.

    Every other link (curated feed entries) is returned as-is with zero
    HTTP calls. If resolution raises for a Google News link, the original,
    unresolved `news.google.com/...` URL is returned instead -- this
    surfaces downstream as `extract_failed` (#20) rather than dropping the
    item or crashing discovery for the whole feed.
    """
    if not _is_google_news_redirect(url):
        return url

    try:
        resolved: str = urlopen(url).geturl()
        return resolved
    except Exception:
        logger.warning("rss: failed to resolve redirect for %s", url, exc_info=True)
        return url
