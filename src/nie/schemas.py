"""Pydantic response models for the web API (issue #34).

New top-level module per `design.md` §15's project layout
(`src/nie/schemas.py`, alongside `config.py`/`db.py`/`models.py`) -- API
response shapes are kept separate from the SQLAlchemy models in
`nie.models`, which describe storage, not the wire format. Only the models
issue #34 needs are defined here; the rest of the API surface (`design.md`
§12) adds its own schemas when those endpoints are built.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class PipelineRunSummary(BaseModel):
    """Summary of one `nie.models.PipelineRun` row, for embedding in
    `WatchStatusResponse.last_run`.

    Mirrors a subset of `PipelineRun`'s columns; deliberately does not
    assert or depend on a particular `stats` shape -- the pipeline runner
    (#18) owns what goes in there.
    """

    id: uuid.UUID
    trigger: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    stats: dict[str, Any]
    error: str | None


class WatchStatusResponse(BaseModel):
    """Response body for `GET /watch/status`, `POST /watch/enable`, and
    `POST /watch/disable` (issue #34).

    `status` is always `"enabled"` or `"disabled"` -- the exact strings
    `nie.models.Watch.status`'s `CheckConstraint` allows. `last_run` is
    `None` when the pipeline has never run for this watch.
    """

    slug: str
    status: str
    last_run: PipelineRunSummary | None
