"""Tests for src/nie/sources/rss.py (`RssProvider`).

Fully offline per _docs/testing-guidelines.md: `feedparser.parse` is given
only local fixture paths (it reads local files with no network access),
and `nie.sources.rss.urlopen` -- the only thing that could make a real
HTTP call -- is always monkeypatched. No test opens a DB session or
depends on execution order with any other test file. `discover()` is
exercised against an in-memory `Watch()` that is never added to a
session or persisted.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nie.models import Watch
from nie.sources import rss
from nie.sources.rss import RssProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sources"
KITCO_FEED = str(FIXTURES_DIR / "kitco_feed.xml")
GOOGLE_NEWS_FEED = str(FIXTURES_DIR / "google_news_silver.xml")


def _watch() -> Watch:
    """An in-memory `Watch`, never persisted -- `RssProvider.discover`
    ignores it entirely, so its field values don't matter here."""
    return Watch(slug="silver-commodity", name="Silver Commodity", status="enabled")


class _FakeResponse:
    """Stand-in for `urlopen`'s return value -- only `.geturl()` is used."""

    def __init__(self, url: str) -> None:
        self._url = url

    def geturl(self) -> str:
        return self._url


def test_discover_parses_kitco_feed_with_direct_links(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_if_called(url: str) -> _FakeResponse:
        raise AssertionError(f"urlopen should not be called for a curated feed, got {url!r}")

    monkeypatch.setattr(rss, "urlopen", _fail_if_called)

    provider = RssProvider(feed_urls=[KITCO_FEED])

    result = list(provider.discover(_watch(), datetime(2020, 1, 1, tzinfo=UTC)))

    assert len(result) == 3

    first = result[0]
    assert first.url == "https://www.kitco.com/news/2026-09-10/silver-breaks-above-30.html"
    assert first.title == "Silver breaks above $30 as industrial demand surges"
    assert first.source_name == "Kitco News"
    assert first.published_at == datetime(2026, 9, 10, 14, 30, 0, tzinfo=UTC)
    assert first.snippet == (
        "Silver prices climbed past $30 an ounce on Thursday as solar panel "
        "manufacturers reported record purchasing."
    )
    assert first.content is None
    assert first.entities is None

    # Every item in this feed keeps its original (unresolved) publisher link
    # and takes source_name from the feed-level title, not a per-item source.
    for item in result:
        assert item.url.startswith("https://www.kitco.com/")
        assert item.source_name == "Kitco News"


def test_discover_resolves_google_news_redirect_via_urlopen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_url = "https://www.reuters.com/markets/commodities/silver-surges-2026-09-12/"
    monkeypatch.setattr(rss, "urlopen", lambda url: _FakeResponse(resolved_url))

    provider = RssProvider(feed_urls=[GOOGLE_NEWS_FEED])

    result = list(provider.discover(_watch(), datetime(2020, 1, 1, tzinfo=UTC)))

    assert len(result) == 3
    first = result[0]
    assert first.url == resolved_url
    assert first.source_name == "Reuters"
    assert "news.google.com" not in first.url


def test_discover_falls_back_to_original_url_when_resolution_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(url: str) -> _FakeResponse:
        raise TimeoutError("simulated resolution failure")

    monkeypatch.setattr(rss, "urlopen", _raise)

    provider = RssProvider(feed_urls=[GOOGLE_NEWS_FEED])

    result = list(provider.discover(_watch(), datetime(2020, 1, 1, tzinfo=UTC)))

    assert len(result) == 3
    for item in result:
        assert item.url.startswith("https://news.google.com/rss/articles/")


def test_discover_skips_unparseable_feed_and_still_returns_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_if_called(url: str) -> _FakeResponse:
        raise AssertionError("urlopen should not be called for the kitco feed")

    monkeypatch.setattr(rss, "urlopen", _fail_if_called)

    provider = RssProvider(
        feed_urls=[str(FIXTURES_DIR / "does-not-exist.xml"), KITCO_FEED]
    )

    result = list(provider.discover(_watch(), datetime(2020, 1, 1, tzinfo=UTC)))

    assert len(result) == 3
    assert all(item.url.startswith("https://www.kitco.com/") for item in result)


def test_discover_ignores_watch_and_since(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rss, "urlopen", lambda url: _FakeResponse("https://example.com/x"))

    provider = RssProvider(feed_urls=[KITCO_FEED, GOOGLE_NEWS_FEED])

    result_early = list(provider.discover(_watch(), datetime(2000, 1, 1, tzinfo=UTC)))
    result_late = list(provider.discover(_watch(), datetime(2099, 1, 1, tzinfo=UTC)))

    other_watch = Watch(slug="other-watch", name="Other Watch", status="enabled")
    result_other_watch = list(provider.discover(other_watch, datetime(2010, 5, 5, tzinfo=UTC)))

    assert result_early == result_late == result_other_watch
