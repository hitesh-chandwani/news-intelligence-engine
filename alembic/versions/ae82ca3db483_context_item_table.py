"""context_item table

Revision ID: ae82ca3db483
Revises: 6cdcd6f2060c
Create Date: 2026-09-11 19:23:16.223787

Creates the `context_item` table (issue #6, `design.md` §4) -- `id` uuid
pk, `watch_id` uuid fk to `watch.id`, `kind` text restricted to
`system`/`user` via a check constraint (not a native Postgres ENUM, same
pattern as `Watch.status`), `label`/`body` text, and `timestamptz`
`created_at`/`updated_at` with `now()` server defaults.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ae82ca3db483"
down_revision: str | None = "6cdcd6f2060c"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "context_item",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("watch_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
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
        sa.CheckConstraint("kind IN ('system', 'user')", name="ck_context_item_kind"),
        sa.ForeignKeyConstraint(["watch_id"], ["watch.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("context_item")
