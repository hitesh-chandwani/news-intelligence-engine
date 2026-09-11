"""event_category table

Revision ID: e0b7a4c2a4f3
Revises: 5896d4dc1a70
Create Date: 2026-09-11 19:54:22.977257

Creates the `event_category` join table (issue #42, `design.md` §4,
FR-011). A pure join table -- no synthetic `id` column, same composite-
primary-key pattern `event_source`/`event_relation` used in #10.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e0b7a4c2a4f3"
down_revision: str | None = "5896d4dc1a70"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "event_category",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("category_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["event.id"]),
        sa.ForeignKeyConstraint(["category_id"], ["category.id"]),
        sa.PrimaryKeyConstraint("event_id", "category_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("event_category")
