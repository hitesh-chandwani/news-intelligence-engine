"""adjudication columns on source

Revision ID: 8b2011b2e4d7
Revises: a0023c0d85d2
Create Date: 2026-09-14 11:05:13.605995

Adds the adjudicate stage's (#25) three new `source` columns --
`adjudication_decision`, `adjudication_event_id`, `adjudication_materiality`
-- and extends `ck_source_status` (#8) with the new `'adjudicated'` value
(adjudication decision recorded, awaiting synthesis, #26's input state).

`adjudication_decision`/`adjudication_materiality` are plain `text` +
`CheckConstraint`, same pattern as every other status/verdict column in
this schema; both stay nullable, `NULL` until `adjudicate_stage` runs.
`adjudication_event_id` is a nullable FK to `event.id`, set only when
`adjudication_decision = 'existing'` (enforced in application code, not a
DB constraint).

`ck_source_status` cannot be altered in place -- Postgres has no
`ALTER CONSTRAINT`, so the old constraint is dropped and the new one
(with `'adjudicated'` added) is created under the same name.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8b2011b2e4d7"
down_revision: str | None = "a0023c0d85d2"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("source", sa.Column("adjudication_decision", sa.Text(), nullable=True))
    op.add_column("source", sa.Column("adjudication_event_id", sa.Uuid(), nullable=True))
    op.add_column("source", sa.Column("adjudication_materiality", sa.Text(), nullable=True))
    op.create_foreign_key(
        "source_adjudication_event_id_fkey",
        "source",
        "event",
        ["adjudication_event_id"],
        ["id"],
    )
    op.drop_constraint("ck_source_status", "source", type_="check")
    op.create_check_constraint(
        "ck_source_status",
        "source",
        "status IN ('discovered', 'extracted', 'extract_failed', "
        "'triaged_out', 'processed', 'adjudicated')",
    )
    op.create_check_constraint(
        "ck_source_adjudication_decision",
        "source",
        "adjudication_decision IN ('new', 'existing', 'noise')",
    )
    op.create_check_constraint(
        "ck_source_adjudication_materiality",
        "source",
        "adjudication_materiality IN ('none', 'minor', 'material')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_source_adjudication_materiality", "source", type_="check")
    op.drop_constraint("ck_source_adjudication_decision", "source", type_="check")
    op.drop_constraint("ck_source_status", "source", type_="check")
    op.create_check_constraint(
        "ck_source_status",
        "source",
        "status IN ('discovered', 'extracted', 'extract_failed', "
        "'triaged_out', 'processed')",
    )
    op.drop_constraint("source_adjudication_event_id_fkey", "source", type_="foreignkey")
    op.drop_column("source", "adjudication_materiality")
    op.drop_column("source", "adjudication_event_id")
    op.drop_column("source", "adjudication_decision")
