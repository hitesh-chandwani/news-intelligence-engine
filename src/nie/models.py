"""SQLAlchemy 2.0 declarative models.

One declarative ``Base`` for the whole app -- every table (this one and
#6-#14) is declared in this module, per the layout in ``_docs/design.md``
§15 (``src/nie/models.py``, not a ``models/`` package). See
``_docs/design.md`` §4 for the source-of-truth schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Text, Uuid, func
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
