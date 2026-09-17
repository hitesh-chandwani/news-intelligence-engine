"""score attempts column on event

Revision ID: df8eaf89f74b
Revises: 811868a6c491
Create Date: 2026-09-17 19:16:52.630244

Adds `event.score_attempts` (#46): a per-row counter of how many times
`score_stage` (#28) has actually called `client.call_structured` for the
row, used to cap retries of `relevance IS NULL` rows rather than
retrying them forever -- the same operational fix #45 established for
`source.extract_attempts`, applied by symmetry to `event`/score.

`Integer`, `nullable=False`, `server_default="0"` -- existing rows
backfill to `0` via the server default, same "add a column to an
existing table" pattern `811868a6c491` (#45) used for
`source.extract_attempts`. No new status/constraint: `event` has no
`status` column at all (unlike `source`), so there is nothing else to
add -- the cap is enforced purely via `Event.score_attempts <
settings.max_score_attempts` in `score_stage`'s selection query.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "df8eaf89f74b"
down_revision: str | None = "811868a6c491"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "event",
        sa.Column("score_attempts", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("event", "score_attempts")
