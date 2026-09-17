"""extract attempts column on source

Revision ID: 811868a6c491
Revises: 8b2011b2e4d7
Create Date: 2026-09-17 17:14:11.312353

Adds `source.extract_attempts` (#45): a per-row counter of how many times
`extract_stage` (#20) has actually attempted an extraction for the row,
used to cap retries of `status="extract_failed"` rows rather than
retrying them forever.

`Integer`, `nullable=False`, `server_default="0"` -- existing rows
backfill to `0` via the server default, same "add columns to an existing
table" pattern `8b2011b2e4d7` (#25) used for the adjudication columns.
No `ck_source_status`/other constraint changes: this is a plain counter
column, not a new status value.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "811868a6c491"
down_revision: str | None = "8b2011b2e4d7"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "source",
        sa.Column("extract_attempts", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("source", "extract_attempts")
