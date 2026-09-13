"""Tests for src/nie/sources/stub.py (`StubProvider`).

Fully offline per _docs/testing-guidelines.md: no network call, no DB
session. `discover()` is exercised against an in-memory `Watch()` that is
never added to a session or persisted, so this file needs no
`session_factory`/`migrated_db` fixture. No test depends on execution
order with any other test file.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from nie.models import Watch
from nie.sources.base import CandidateItem
from nie.sources.stub import StubProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sources"


def _watch() -> Watch:
    """An in-memory `Watch`, never persisted -- `StubProvider.discover`
    ignores it entirely, so its field values don't matter here."""
    return Watch(slug="silver-commodity", name="Silver Commodity", status="enabled")


# Expected `CandidateItem`s, values taken from tests/fixtures/sources/*.json,
# in filename-sorted order.
EXPECTED_ITEMS = [
    CandidateItem(
        url="https://example.com/silver-etf-inflows",
        title="Silver ETF Inflows Surge to Record High",
        source_name="Kitco News",
        published_at=datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC),
        snippet="Silver ETF holdings rose sharply this week amid strong investor demand.",
        entities=["Silver", "ETF", "Kitco"],
    ),
    CandidateItem(
        url="https://example.com/mining-output-report",
        title="Global Silver Mining Output Report Released",
        source_name="Mining.com",
        published_at=datetime(2026, 9, 2, 8, 30, 0, tzinfo=UTC),
        snippet=(
            "The latest report details silver mining output trends across "
            "major producing nations."
        ),
    ),
    CandidateItem(
        url="https://example.com/fed-rate-decision-silver",
        title="Fed Rate Decision Sends Silver Prices Higher",
        source_name="Reuters",
        published_at=datetime(2026, 9, 3, 15, 45, 0, tzinfo=UTC),
        snippet="Silver prices jumped after the Federal Reserve signaled a pause in rate hikes.",
        content=(
            "The Federal Reserve's decision to hold interest rates steady sent silver "
            "prices higher in Tuesday trading, as investors sought inflation hedges amid "
            "signals of a dovish pivot."
        ),
    ),
]


def test_discover_returns_expected_items_in_filename_order() -> None:
    provider = StubProvider(fixtures_dir=FIXTURES_DIR)

    result = list(provider.discover(_watch(), datetime(2020, 1, 1, tzinfo=UTC)))

    assert result == EXPECTED_ITEMS


def test_discover_ignores_since() -> None:
    provider = StubProvider(fixtures_dir=FIXTURES_DIR)

    result_early = list(provider.discover(_watch(), datetime(2000, 1, 1, tzinfo=UTC)))
    result_late = list(provider.discover(_watch(), datetime(2099, 1, 1, tzinfo=UTC)))

    assert result_early == EXPECTED_ITEMS
    assert result_late == EXPECTED_ITEMS


def test_discover_raises_value_error_naming_file_and_field_on_missing_field(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "bad.json"
    fixture.write_text(
        '{"title": "Missing URL", "source_name": "Test", '
        '"published_at": "2026-01-01T00:00:00+00:00", "snippet": "..."}'
    )
    provider = StubProvider(fixtures_dir=tmp_path)

    with pytest.raises(ValueError, match="bad.json") as exc_info:
        list(provider.discover(_watch(), datetime(2020, 1, 1, tzinfo=UTC)))

    assert "url" in str(exc_info.value)


def test_discover_yields_nothing_for_empty_directory(tmp_path: Path) -> None:
    provider = StubProvider(fixtures_dir=tmp_path)

    result = list(provider.discover(_watch(), datetime(2020, 1, 1, tzinfo=UTC)))

    assert result == []
