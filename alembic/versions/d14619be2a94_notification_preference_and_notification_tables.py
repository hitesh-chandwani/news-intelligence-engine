"""notification_preference and notification tables

Revision ID: d14619be2a94
Revises: e0b7a4c2a4f3
Create Date: 2026-09-11 20:02:01.043792

Creates the `notification_preference` and `notification` tables (issue
#11, `design.md` §4, FR-020 through FR-024).

`notification_preference.watch_id` is the primary key itself -- no
synthetic `id` column, unlike the `Watch`/`Source`/`Event` synthetic-id
pattern -- because `design.md` §4's column list for this table has no
`id`, and making the foreign key the primary key is what enforces "one
row per watch". `categories`/`channels`/`channels_sent` are Postgres
`ARRAY(Text)`, not `JSONB` -- flat string lists, unlike the structured
`notification.payload` JSONB column.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d14619be2a94"
down_revision: str | None = "e0b7a4c2a4f3"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "notification_preference",
        sa.Column("watch_id", sa.Uuid(), nullable=False),
        sa.Column("min_importance", sa.Text(), nullable=False),
        sa.Column("categories", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("channels", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.CheckConstraint(
            "min_importance IN ('low', 'medium', 'high', 'critical')",
            name="ck_notification_preference_min_importance",
        ),
        sa.ForeignKeyConstraint(["watch_id"], ["watch.id"]),
        sa.PrimaryKeyConstraint("watch_id"),
    )
    op.create_table(
        "notification",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("watch_id", sa.Uuid(), nullable=False),
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("channels_sent", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("read_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "reason IN ('new-event', 'material-update')",
            name="ck_notification_reason",
        ),
        sa.ForeignKeyConstraint(["event_id"], ["event.id"]),
        sa.ForeignKeyConstraint(["watch_id"], ["watch.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("notification")
    op.drop_table("notification_preference")
