"""category table

Revision ID: f0a9e574ee03
Revises: ae82ca3db483
Create Date: 2026-09-11 19:28:00.596915

Creates the `category` table (issue #7, `design.md` §4) -- `id` uuid pk,
`slug` unique text, `name` text. Unlike `watch`/`context_item`, this table
has no `created_at`/`updated_at` and no FK to `watch`: it's a small, global,
seeded lookup table, not a per-watch table users write rows into. Seeded via
`nie.seed.categories.seed_categories`, wired into the seed script by #14.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f0a9e574ee03"
down_revision: str | None = "ae82ca3db483"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "category",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("category")
