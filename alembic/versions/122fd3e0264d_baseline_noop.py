"""baseline noop

Revision ID: 122fd3e0264d
Revises:
Create Date: 2026-09-11 19:11:55.975981

No schema changes -- this migration only proves the async engine + Alembic
plumbing (connect, upgrade, downgrade) works end to end against the Docker
Compose Postgres. Real tables start at issue #5.
"""

from __future__ import annotations

# revision identifiers, used by Alembic.
revision: str = "122fd3e0264d"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema. Intentionally a no-op."""


def downgrade() -> None:
    """Downgrade schema. Intentionally a no-op."""
