"""Tests for src/nie/sources/extract_trafilatura.py (`TrafilaturaExtractor`).

Fully offline per _docs/testing-guidelines.md: `trafilatura.fetch_url` is
monkeypatched in every test, so no test makes a live network call, opens
a DB session, or depends on execution order with any other test file.
Only the fetch is stubbed -- the real (un-mocked) `trafilatura.
bare_extraction` parsing path runs against the fixture HTML in the happy
path and the parse-failure path.
"""

from pathlib import Path

import pytest

from nie.sources import extract_trafilatura
from nie.sources.base import ExtractionError
from nie.sources.extract_trafilatura import TrafilaturaExtractor

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sources" / "silver-etf-inflows-article.html"
URL = "https://example.com/silver-etf-inflows"


def test_extract_returns_title_and_text_from_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    html = FIXTURE_PATH.read_text()
    monkeypatch.setattr(
        extract_trafilatura.trafilatura, "fetch_url", lambda url, config=None: html
    )

    result = TrafilaturaExtractor().extract(URL)

    assert result.title == "Silver ETF Inflows Surge to Record High"
    assert "largest weekly inflow on record" in result.text
    assert "central bank policy meeting" in result.text


def test_extract_raises_extraction_error_naming_url_on_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        extract_trafilatura.trafilatura, "fetch_url", lambda url, config=None: None
    )

    with pytest.raises(ExtractionError, match=URL):
        TrafilaturaExtractor().extract(URL)


def test_extract_raises_extraction_error_naming_url_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        extract_trafilatura.trafilatura,
        "fetch_url",
        lambda url, config=None: "<html><body></body></html>",
    )

    with pytest.raises(ExtractionError, match=URL):
        TrafilaturaExtractor().extract(URL)
