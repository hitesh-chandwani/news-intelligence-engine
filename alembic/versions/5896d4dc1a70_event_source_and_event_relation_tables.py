"""event_source and event_relation tables

Revision ID: 5896d4dc1a70
Revises: 504e128202fd
Create Date: 2026-09-11 19:50:15.337984

Creates the `event_source` and `event_relation` join tables (issue #10,
`design.md` §4, FR-009, FR-018). Both are pure join tables -- no synthetic
`id` column, composite primary keys instead. `event_relation.relation` is
plain text restricted via a check constraint (same pattern as every other
status/verdict column), plus a second check constraint rejecting
`from_event_id == to_event_id`.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5896d4dc1a70"
down_revision: str | None = "504e128202fd"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "event_relation",
        sa.Column("from_event_id", sa.Uuid(), nullable=False),
        sa.Column("to_event_id", sa.Uuid(), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "relation IN ('precedes', 'similar', 'escalation-of', 'context-for')",
            name="ck_event_relation_relation",
        ),
        sa.CheckConstraint(
            "from_event_id != to_event_id", name="ck_event_relation_no_self_relation"
        ),
        sa.ForeignKeyConstraint(["from_event_id"], ["event.id"]),
        sa.ForeignKeyConstraint(["to_event_id"], ["event.id"]),
        sa.PrimaryKeyConstraint("from_event_id", "to_event_id", "relation"),
    )
    op.create_table(
        "event_source",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column(
            "linked_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["event_id"], ["event.id"]),
        sa.ForeignKeyConstraint(["source_id"], ["source.id"]),
        sa.PrimaryKeyConstraint("event_id", "source_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("event_source")
    op.drop_table("event_relation")
