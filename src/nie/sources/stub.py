"""`StubProvider` -- reads fixture JSON files instead of the network.

Per `_docs/design.md` §7 build order item 1: lets the whole pipeline run
offline in development and CI, with no live discovery, DB, or network
access needed. Wiring `DISCOVERY_PROVIDERS` config to instantiate this
(and pointing `fixtures_dir` at `tests/fixtures/sources/` in dev/CI) is
#18, not this module's job.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from nie.models import Watch
from nie.sources.base import CandidateItem

_REQUIRED_FIELDS = ("url", "title", "source_name", "published_at", "snippet")


class StubProvider:
    """Reads every `*.json` file directly under `fixtures_dir` (non-
    recursive), sorted by filename for deterministic order, into one
    `CandidateItem` each.

    `fixtures_dir` is a required constructor argument with no default
    that reaches from `src/` into `tests/` -- whatever wires providers
    together (#18) is responsible for pointing it at
    `tests/fixtures/sources/` in dev/CI.
    """

    name = "stub"

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = fixtures_dir

    def discover(self, watch: Watch, since: datetime) -> Iterable[CandidateItem]:
        """Yield one `CandidateItem` per `*.json` fixture file.

        Both `watch` and `since` are ignored: the stub always returns the
        same fixtures on every call, regardless of which watch or `since`
        timestamp is passed.
        """
        for path in sorted(self.fixtures_dir.glob("*.json")):
            yield _parse_fixture(path)


def _parse_fixture(path: Path) -> CandidateItem:
    data: dict[str, Any] = json.loads(path.read_text())

    for field in _REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"{path.name}: missing required field {field!r}")

    return CandidateItem(
        url=data["url"],
        title=data["title"],
        source_name=data["source_name"],
        published_at=datetime.fromisoformat(data["published_at"]),
        snippet=data["snippet"],
        content=data.get("content"),
        entities=data.get("entities"),
    )
