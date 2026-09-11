"""watch table

Revision ID: 6cdcd6f2060c
Revises: 122fd3e0264d
Create Date: 2026-09-11 19:17:47.097893

Creates the `watch` table (issue #5, `design.md` §4, FR-001/FR-002) --
`id` uuid pk, `slug` unique text, `name` text, `status` text restricted to
`enabled`/`disabled` via a check constraint (not a native Postgres ENUM),
and `timestamptz` `created_at`/`updated_at` with `now()` server defaults.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6cdcd6f2060c"
down_revision: str | None = "122fd3e0264d"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "watch",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
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
        sa.CheckConstraint("status IN ('enabled', 'disabled')", name="ck_watch_status"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("watch")
