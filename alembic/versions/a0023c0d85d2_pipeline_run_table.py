"""pipeline_run table

Revision ID: a0023c0d85d2
Revises: 1b0347ae85e5
Create Date: 2026-09-11 20:13:21.111902

Creates the `pipeline_run` table (issue #13, `design.md` §4
"observability", §5, §12). Deliberately has no `watch_id` column and no
foreign key -- a `pipeline_run` row is an operational record of one
execution of the whole pipeline process, not domain data scoped to a
`Watch`. `trigger`/`status` are plain `text` + `CheckConstraint`, same
pattern as every other status/verdict column in this schema.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a0023c0d85d2"
down_revision: str | None = "1b0347ae85e5"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "pipeline_run",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("trigger", sa.Text(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("stats", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "trigger IN ('schedule', 'manual')", name="ck_pipeline_run_trigger"
        ),
        sa.CheckConstraint(
            "status IN ('running', 'ok', 'partial', 'failed')",
            name="ck_pipeline_run_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("pipeline_run")
