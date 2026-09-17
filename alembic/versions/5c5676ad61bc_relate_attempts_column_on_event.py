"""relate attempts column on event

Revision ID: 5c5676ad61bc
Revises: 16a090affbd7
Create Date: 2026-09-18 02:18:24.397613

Adds `event.relate_attempts` (#55): a per-row counter of how many times
`relate_stage` (#29) has actually called `client.call_structured` for the
row, used to cap retries of a genuine LLM-call failure (a
`json.JSONDecodeError`/`pydantic.ValidationError` still raised after
`call_structured`'s own internal validate-then-retry-once) rather than
retrying it forever -- the same operational fix #45/#46 established for
`source.extract_attempts`/`event.score_attempts`, applied by symmetry to
`event`/relate. Distinct from `event.related_at` (#47), which caps
retrying a *success* (a legitimately empty `RelationSet`) -- this column
caps retrying a *failure*.

`Integer`, `nullable=False`, `server_default="0"` -- existing rows
backfill to `0` via the server default, same "add a column to an
existing table" pattern `df8eaf89f74b` (#46) used for
`event.score_attempts`. No data backfill needed, unlike `42abf5f1004e`'s
`related_at` migration: every existing row correctly starts at
attempt-count `0`.

`down_revision` is `16a090affbd7` (#53, `add_cancelled_to_pipeline_run_status`)
-- the current migration head as of this issue's grooming, confirmed via
`alembic heads` before writing this file. Not `df8eaf89f74b` (#46): two
migrations (`42abf5f1004e` for #47, then `16a090affbd7` for #53) have
since landed on top of it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5c5676ad61bc"
down_revision: str | None = "16a090affbd7"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "event",
        sa.Column("relate_attempts", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("event", "relate_attempts")
