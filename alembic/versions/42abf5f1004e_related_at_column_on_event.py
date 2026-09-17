"""related_at column on event

Revision ID: 42abf5f1004e
Revises: df8eaf89f74b
Create Date: 2026-09-17 22:30:18.420569

Adds `event.related_at` (#47): a nullable completion marker set once
`relate_stage` (#29) has successfully produced a parsed `RelationSet`
response for the row, regardless of how many relations survived
filtering (including zero) -- mirroring `Source.extracted_at`'s "set
once on a successful pass, never re-checked" semantics, not a
`extract_attempts`/`score_attempts`-style retry-cap counter (#45/#46).

`DateTime(timezone=True)`, `nullable=True`, no default -- unlike
`811868a6c491`/`df8eaf89f74b`'s plain `server_default="0"` counter
columns, a blanket default here would be wrong: `relate_stage`'s
*old* selection criterion ("no existing outbound `event_relation`
row") already considered some rows done, and those rows must not be
reselected and re-sent to the LLM just because the selection criterion
changed. So this migration backfills in two different ways via
`op.execute(...)`, not the column default:

- Any `event` row that already has >=1 outbound `event_relation` row
  (i.e. was already "done" under the old `~exists(...)` criterion)
  gets `related_at` set to `now()`.
- Every other row is left `NULL` (`ADD COLUMN` with no default already
  leaves existing rows `NULL`, so there is nothing to do for this
  half -- it's listed here for clarity, not as a second statement).

Without the first backfill, every event already correctly related
before this migration would be reselected and re-sent to the LLM
exactly once right after deploy, purely because the selection
criterion changed out from under it.

`EventRelation` (#10) has no created-at-style column of its own to
reuse for the backfilled timestamp (only `from_event_id`,
`to_event_id`, `relation`, `rationale`), so `now()` is used directly,
same as the issue's own suggested fallback.

Downgrade only drops the column -- there is no need to reverse the
backfill logic, same "downgrade is a plain structural inverse" precedent
`811868a6c491`/`df8eaf89f74b` set for their own counter columns.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "42abf5f1004e"
down_revision: str | None = "df8eaf89f74b"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("event", sa.Column("related_at", sa.DateTime(timezone=True), nullable=True))
    op.execute(
        """
        UPDATE event
        SET related_at = now()
        WHERE related_at IS NULL
          AND EXISTS (
              SELECT 1 FROM event_relation
              WHERE event_relation.from_event_id = event.id
          )
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("event", "related_at")
