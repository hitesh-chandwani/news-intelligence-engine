"""SQLAlchemy 2.0 declarative models.

One declarative ``Base`` for the whole app -- every table (this one and
#6-#14) is declared in this module, per the layout in ``_docs/design.md``
§15 (``src/nie/models.py``, not a ``models/`` package). See
``_docs/design.md`` §4 for the source-of-truth schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Text, UniqueConstraint, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Watch(Base):
    """A monitored topic (`design.md` §4, FR-001, FR-002).

    ``status`` is plain ``text`` with a ``CheckConstraint`` rather than a
    native Postgres ``ENUM`` -- `design.md` §4 types every status/verdict
    column across the schema as ``text``, and this model sets that
    precedent for #7-#9.
    """

    __tablename__ = "watch"
    __table_args__ = (
        CheckConstraint("status IN ('enabled', 'disabled')", name="ck_watch_status"),
    )

    # Python-side default (uuid.uuid4), not a Postgres server-side default --
    # keeps id generation DB-agnostic and testable without hitting the DB.
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Source(Base):
    """A discovered piece of web content and its processing status
    (`design.md` §4, FR-007), keyed to the `Watch` that discovered it.

    ``status`` is plain ``text`` with a ``CheckConstraint`` rather than a
    native Postgres ``ENUM``, same pattern as `Watch.status`/
    `ContextItem.kind`. The `(watch_id, url)` pair is unique -- dedup is
    per-watch, not global, so the same URL can be discovered by two
    different watches. ``embedding`` is populated later by #21 (local
    embeddings module + embed stage), not by this task.
    """

    __tablename__ = "source"
    __table_args__ = (
        UniqueConstraint("watch_id", "url", name="uq_source_watch_id_url"),
        CheckConstraint(
            "status IN ('discovered', 'extracted', 'extract_failed', "
            "'triaged_out', 'processed')",
            name="ck_source_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    watch_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("watch.id"), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    content: Mapped[str | None] = mapped_column(Text)
    extracted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    entities: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))
    status: Mapped[str] = mapped_column(Text, nullable=False)
    triage_note: Mapped[str | None] = mapped_column(Text)


class Category(Base):
    """A category in the event category taxonomy (`design.md` §4, #7).

    A small, global, seeded lookup table -- unlike `Watch`/`ContextItem` it
    has no `created_at`/`updated_at` and no `watch_id` (`design.md` §4 lists
    only `id`, `slug`, `name` for `category`). Rows are inserted by
    `nie.seed.categories.seed_categories`, not created by users.
    """

    __tablename__ = "category"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)


class Event(Base):
    """An identified event: fact/interpretation split, scoring, entities,
    and embedding (`design.md` §4, FR-010, FR-012, FR-014, FR-015, FR-016,
    FR-019), keyed to the `Watch` that surfaced it. The most complex table
    in the schema so far.

    ``fact_summary`` (observed information only) and ``interpretation``
    (system interpretation) are kept in schema-separate columns per
    FR-016. ``relevance``/``importance``/``impact_direction``/
    ``impact_confidence`` are plain ``text`` + ``CheckConstraint``, same
    pattern as `Watch.status`/`Source.status`, but -- unlike those -- are
    nullable: pipeline stage 7 (synthesize, #26) creates the row before
    stage 8 (score, #28) fills these in, so they start out ``NULL``.
    Postgres ``IN (...)`` checks pass ``NULL`` through automatically, so no
    extra ``OR col IS NULL`` clause is needed. ``impact_reason`` is free
    text with no CheckConstraint (not enum-like).

    ``embedding`` is **not** nullable, unlike `Source.embedding` (#8) --
    stage 7 builds the full event record (title, fact_summary/
    interpretation, event_date, entities, embedding) in one shot, with the
    embedding needed immediately for stage 5's vector match on future runs.

    ``last_material_update_at`` has a ``server_default`` but deliberately
    **no** ``onupdate`` (unlike ``updated_at``) -- it's set once at insert
    and only bumped by application code on a material change (FR-019),
    independent of ``updated_at`` which auto-bumps on every change.
    """

    __tablename__ = "event"
    __table_args__ = (
        CheckConstraint(
            "relevance IN ('irrelevant', 'low', 'medium', 'high')",
            name="ck_event_relevance",
        ),
        CheckConstraint(
            "importance IN ('low', 'medium', 'high', 'critical')",
            name="ck_event_importance",
        ),
        CheckConstraint(
            "impact_direction IN ('bullish', 'bearish', 'neutral', 'unclear')",
            name="ck_event_impact_direction",
        ),
        CheckConstraint(
            "impact_confidence IN ('low', 'medium', 'high')",
            name="ck_event_impact_confidence",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    watch_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("watch.id"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    fact_summary: Mapped[str] = mapped_column(Text, nullable=False)
    interpretation: Mapped[str] = mapped_column(Text, nullable=False)
    event_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    relevance: Mapped[str | None] = mapped_column(Text)
    importance: Mapped[str | None] = mapped_column(Text)
    impact_direction: Mapped[str | None] = mapped_column(Text)
    impact_reason: Mapped[str | None] = mapped_column(Text)
    impact_confidence: Mapped[str | None] = mapped_column(Text)
    entities: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    last_material_update_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class ContextItem(Base):
    """Background context attached to a `Watch` (`design.md` §4, #6).

    Holds both seeded Silver background context (`kind='system'`, content
    loaded later by #14's seed script) and user-provided context
    (`kind='user'`, written later by #35's context editor). Like
    `Watch.status`, `kind` is plain ``text`` with a ``CheckConstraint``
    rather than a native Postgres ``ENUM``.
    """

    __tablename__ = "context_item"
    __table_args__ = (
        CheckConstraint("kind IN ('system', 'user')", name="ck_context_item_kind"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    watch_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("watch.id"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
