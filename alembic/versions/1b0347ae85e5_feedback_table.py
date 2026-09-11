"""feedback table

Revision ID: 1b0347ae85e5
Revises: d14619be2a94
Create Date: 2026-09-11 20:07:21.135530

Creates the `feedback` table (issue #12, `design.md` §4, FR-025, FR-026).

`event_id` is `nullable=False` -- every feedback row is against an event,
since the only write path is `POST /events/{id}/feedback` (`design.md`
§11/§12); there is no `notification_id` column. `verdict` is plain `text`
+ `CheckConstraint`, same pattern as `notification.reason`/every other
status/verdict column in this schema.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1b0347ae85e5"
down_revision: str | None = "d14619be2a94"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("watch_id", sa.Uuid(), nullable=False),
        sa.Column("event_id", sa.Uuid(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "verdict IN ('useful', 'not_useful', 'too_many_similar', "
            "'more_like_this', 'less_of_this')",
            name="ck_feedback_verdict",
        ),
        sa.ForeignKeyConstraint(["event_id"], ["event.id"]),
        sa.ForeignKeyConstraint(["watch_id"], ["watch.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("feedback")
