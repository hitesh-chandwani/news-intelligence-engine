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
    """Downgrade schema.

    Unlike a plain `ADD COLUMN` migration's downgrade (e.g. `811868a6c491`/
    `df8eaf89f74b`/`42abf5f1004e`, always safe to drop regardless of data),
    this re-narrows an existing `CheckConstraint` -- Postgres will correctly
    refuse this downgrade (`CheckViolationError`) once any `pipeline_run`
    row has `status = 'cancelled'`, e.g. from the cancel endpoint (#53) or
    from this repo's own test suite creating one against the shared,
    never-truncated Compose Postgres. That failure is the DB doing its job,
    not a bug in this migration -- downgrading below a constraint value
    real data already uses is inherently unsafe, and there is no data
    transformation here (unlike a column drop) that could make it safe.
    """
    op.drop_constraint("ck_pipeline_run_status", "pipeline_run", type_="check")
    op.create_check_constraint(
        "ck_pipeline_run_status",
        "pipeline_run",
        "status IN ('running', 'ok', 'partial', 'failed')",
    )
