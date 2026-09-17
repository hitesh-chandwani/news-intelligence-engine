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
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class ContextItemResponse(BaseModel):
    """Response body for a single `nie.models.ContextItem` row (issue #35).

    Mirrors every column except `watch_id` -- every endpoint in
    `nie.web.routers.context` is scoped to the single Silver watch
    (`nie.seed.run.SILVER_WATCH_SLUG`), so the watch a row belongs to is
    never ambiguous and not worth echoing back on the wire.
    """

    id: uuid.UUID
    kind: str
    label: str
    body: str
    created_at: datetime
    updated_at: datetime


class ContextItemCreate(BaseModel):
    """Request body for `POST /context` (issue #35).

    Deliberately has **no** `kind` field -- the router always creates
    `kind="user"` rows, so a client can never request `kind="system"`
    through this schema. `label`/`body` are both required and must be
    non-blank (whitespace-only strings rejected), producing FastAPI's
    standard `422` on a missing/blank field.
    """

    label: str = Field(min_length=1)
    body: str = Field(min_length=1)

    @field_validator("label", "body")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ContextItemUpdate(BaseModel):
    """Request body for `PATCH /context/{id}` (issue #35).

    Both fields are optional -- the router updates only whichever fields
    are set -- but a model validator rejects a body where both are `None`
    (`422`), since a no-op PATCH is a client error, not a valid request.
    """

    label: str | None = None
    body: str | None = None

    @model_validator(mode="after")
    def _at_least_one_set(self) -> ContextItemUpdate:
        if self.label is None and self.body is None:
            raise ValueError("at least one of label or body must be set")
        return self


# The exact value sets `nie.models.NotificationPreference`'s
# `CheckConstraint`s (`min_importance`) and `design.md` §10 (`channels`)
# allow -- shared by both schemas below so the two `field_validator`s stay
# in lockstep with each other and with the DB constraint.
_MIN_IMPORTANCE_VALUES = {"low", "medium", "high", "critical"}
_CHANNEL_VALUES = {"telegram", "email", "inapp"}


class PreferenceResponse(BaseModel):
    """Response body for `GET /preferences` and `PATCH /preferences`
    (issue #36). Mirrors every column of a `nie.models.NotificationPreference`
    row, including `watch_id` -- unlike `ContextItemResponse`, there's no
    established precedent yet for omitting it, and this table's primary key
    *is* `watch_id` (no synthetic `id` column).
    """

    watch_id: uuid.UUID
    min_importance: str
    categories: list[str]
    channels: list[str]


class PreferenceUpdate(BaseModel):
    """Request body for `PATCH /preferences` (issue #36).

    All three fields are optional -- the router updates only whichever
    fields are set, leaving the rest unchanged. Unlike `ContextItemUpdate`,
    there is **no** model validator rejecting an all-`None` body: a `PATCH`
    that sets nothing is a harmless no-op here, since every field already
    has a well-defined current value to leave alone.

    `categories` is deliberately **not** validated against the live
    `category` table -- a category slug deleted after being selected would
    otherwise 422 forever, and the notify gate (#30) already treats an
    unrecognized slug in `pref.categories` as simply never matching.
    """

    min_importance: str | None = None
    categories: list[str] | None = None
    channels: list[str] | None = None

    @field_validator("categories", "channels", mode="before")
    @classmethod
    def _normalize_checkbox_group(cls, value: Any) -> Any:
        """Reshape what htmx's `json-enc` extension actually produces from
        a checkbox group into the plain list `_valid_channels`/Pydantic's
        own list coercion expect, before that validation runs (QA's #36
        FAIL comment -- same root-cause class as #35's form-encoding bug,
        caught only because QA simulated real checkbox serialization
        instead of passing a clean JSON list directly):

        - htmx only arrays a `name` when *two or more* elements share it
          (`addValueToValues` in `htmx.js`); a single checked checkbox in
          the group serializes as a bare string (e.g.
          `{"channels": "telegram"}`), not a one-element list. Wrap a bare
          string into `[value]` so it validates the same as a real
          multi-select.
        - htmx omits an unchecked checkbox from the submitted values
          entirely, so `partials/preferences_form.html`'s fieldsets each
          carry a hidden `<input type="hidden" name="..." value="">`
          fallback ahead of the checkboxes to keep the key present (and
          therefore not silently no-op the PATCH) even when nothing in
          the group is checked. Drop empty-string entries from a list so
          that fallback -- alone, or mixed with real checked values --
          round-trips to `[]` when nothing is checked and to just the
          real selections otherwise.

        `None` (the field not sent at all -- a genuine "leave unchanged"
        PATCH) passes through unchanged; this only reshapes a value that
        was actually sent.
        """
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return [item for item in value if item != ""]
        return value

    @field_validator("min_importance")
    @classmethod
    def _valid_min_importance(cls, value: str | None) -> str | None:
        if value is not None and value not in _MIN_IMPORTANCE_VALUES:
            raise ValueError(f"min_importance must be one of {sorted(_MIN_IMPORTANCE_VALUES)}")
        return value

    @field_validator("channels")
    @classmethod
    def _valid_channels(cls, value: list[str] | None) -> list[str] | None:
        if value is not None:
            invalid = sorted(set(value) - _CHANNEL_VALUES)
            if invalid:
                raise ValueError(f"channels entries must be one of {sorted(_CHANNEL_VALUES)}")
        return value


class EventSummary(BaseModel):
    """One row of `GET /events` (issue #37): the subset of
    `nie.models.Event` needed for a timeline row, plus its category slugs
    (via `EventCategory`/`Category`) -- same "slug list, no join object"
    convention `PreferenceResponse.categories` (#36) set. Does not include
    `fact_summary`/`interpretation`/`entities`/`sources`/`related_events` --
    those are `EventDetail`-only, fetched via `GET /events/{id}`.
    """

    id: uuid.UUID
    title: str
    event_date: datetime | None
    discovered_at: datetime
    relevance: str | None
    importance: str | None
    impact_direction: str | None
    impact_confidence: str | None
    categories: list[str]


class EventSourceRef(BaseModel):
    """One `nie.models.Source` linked to an event via `EventSource`, as
    embedded in `EventDetail.sources` (issue #37). Carries `url` so the UI
    can link out to the original article (FR-022).
    """

    id: uuid.UUID
    url: str
    title: str
    source_name: str
    published_at: datetime | None


class RelatedEventRef(BaseModel):
    """One `nie.models.EventRelation` row joined to the *other* event's
    `title` (issue #37), as embedded in `EventDetail.related_events`.

    `direction="outgoing"` when the event being described is
    `from_event_id` (e.g. `relation="precedes"` reads "this event precedes
    `title`"); `direction="incoming"` when it is `to_event_id` (reads
    "`title` precedes this event").
    """

    event_id: uuid.UUID
    title: str
    relation: str
    direction: Literal["outgoing", "incoming"]


class NotificationResponse(BaseModel):
    """Response body for `GET /notifications` (array) and
    `POST /notifications/{id}/read` (single object) (issue #38). Mirrors
    every column of a `nie.models.Notification` row, including `watch_id`
    (unlike `ContextItemResponse`'s omission of it for #35) -- unlike the
    context/preferences/events routers, this one's `POST` targets a
    notification by its own globally-unique `id` with no Silver-watch
    scoping in the lookup, so echoing `watch_id` back is useful context.

    `payload` is `dict[str, Any]`, not re-validated against
    `nie.pipeline.notify.NotificationPayload` here -- by the time a
    `Notification` row exists, `payload` is already-validated JSONB (that
    Pydantic model's own `model_dump()`, per #30/#48), so this schema just
    passes it through. `reason` is narrowed to the exact two string values
    `nie.models.Notification`'s `CheckConstraint` allows, same
    "plain `text` column, `Literal` on the wire" pattern used throughout
    this module.
    """

    id: uuid.UUID
    watch_id: uuid.UUID
    event_id: uuid.UUID
    reason: Literal["new-event", "material-update"]
    payload: dict[str, Any]
    channels_sent: list[str]
    created_at: datetime
    read_at: datetime | None


class EventDetail(BaseModel):
    """Full response body for `GET /events/{id}` (issue #37): every column
    of a `nie.models.Event` row (`fact_summary`/`interpretation` kept as
    distinct fields, per FR-016), plus its category slugs, linked sources,
    and related events in both relation directions.
    """

    id: uuid.UUID
    title: str
    fact_summary: str
    interpretation: str
    event_date: datetime | None
    discovered_at: datetime
    relevance: str | None
    importance: str | None
    impact_direction: str | None
    impact_reason: str | None
    impact_confidence: str | None
    entities: list[str]
    categories: list[str]
    sources: list[EventSourceRef]
    related_events: list[RelatedEventRef]
