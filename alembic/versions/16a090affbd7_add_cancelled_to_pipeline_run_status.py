"""add cancelled to pipeline_run status

Revision ID: 16a090affbd7
Revises: 42abf5f1004e
Create Date: 2026-09-18 00:00:00.000000

Widens `ck_pipeline_run_status` (`src/nie/models.py`,
`alembic/versions/a0023c0d85d2_pipeline_run_table.py`) to allow the new
`'cancelled'` terminal status (issue #53), used by the new
`POST /pipeline/runs/{run_id}/cancel` endpoint and by `run_pipeline`'s
now-conditional closing write (`src/nie/pipeline/runner.py`) when it
finds the row has already been cancelled out from under it.

Same "no `ALTER CONSTRAINT` in Postgres, drop and recreate under the
same name" pattern
`alembic/versions/8b2011b2e4d7_adjudication_columns_on_source.py` used
for `ck_source_status`.
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "16a090affbd7"
down_revision: str | None = "42abf5f1004e"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint("ck_pipeline_run_status", "pipeline_run", type_="check")
    op.create_check_constraint(
        "ck_pipeline_run_status",
        "pipeline_run",
        "status IN ('running', 'ok', 'partial', 'failed', 'cancelled')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_pipeline_run_status", "pipeline_run", type_="check")
    op.create_check_constraint(
        "ck_pipeline_run_status",
        "pipeline_run",
        "status IN ('running', 'ok', 'partial', 'failed')",
    )
