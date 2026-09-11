"""event table

Revision ID: 504e128202fd
Revises: 6e4f17b2d041
Create Date: 2026-09-11 19:44:14.566387

Creates the `event` table (issue #9, `design.md` §4, FR-010, FR-012,
FR-014, FR-015, FR-016, FR-019) -- the fact/interpretation split, the
relevance/importance/impact_* scoring fields, entities, a 384-dim
embedding, and `last_material_update_at`, keyed to the `watch` that
surfaced it. `relevance`/`importance`/`impact_direction`/
`impact_confidence` are plain text restricted via check constraints (same
pattern as `watch.status`/`source.status`) but, unlike those, nullable --
the synthesize stage (#26) creates the row before the score stage (#28)
fills these in.

Does **not** re-run `CREATE EXTENSION IF NOT EXISTS vector` -- already
done in `6e4f17b2d041` (#8).
"""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "504e128202fd"
down_revision: str | None = "6e4f17b2d041"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "event",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("watch_id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("fact_summary", sa.Text(), nullable=False),
        sa.Column("interpretation", sa.Text(), nullable=False),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "discovered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("relevance", sa.Text(), nullable=True),
        sa.Column("importance", sa.Text(), nullable=True),
        sa.Column("impact_direction", sa.Text(), nullable=True),
        sa.Column("impact_reason", sa.Text(), nullable=True),
        sa.Column("impact_confidence", sa.Text(), nullable=True),
        sa.Column("entities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(384), nullable=False),
        sa.Column(
            "last_material_update_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "relevance IN ('irrelevant', 'low', 'medium', 'high')",
            name="ck_event_relevance",
        ),
        sa.CheckConstraint(
            "importance IN ('low', 'medium', 'high', 'critical')",
            name="ck_event_importance",
        ),
        sa.CheckConstraint(
            "impact_direction IN ('bullish', 'bearish', 'neutral', 'unclear')",
            name="ck_event_impact_direction",
        ),
        sa.CheckConstraint(
            "impact_confidence IN ('low', 'medium', 'high')",
            name="ck_event_impact_confidence",
        ),
        sa.ForeignKeyConstraint(["watch_id"], ["watch.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("event")
