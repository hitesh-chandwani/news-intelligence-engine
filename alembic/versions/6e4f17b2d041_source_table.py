"""source table

Revision ID: 6e4f17b2d041
Revises: f0a9e574ee03
Create Date: 2026-09-11 19:34:42.594181

Creates the `source` table (issue #8, `design.md` §4, FR-007) -- a
discovered piece of web content and its processing status, keyed to the
`watch` that discovered it. `status` is plain text restricted via a check
constraint (not a native Postgres ENUM, same pattern as `watch.status`/
`context_item.kind`), `(watch_id, url)` is uniquely constrained per-watch
(not globally), and `embedding` is a 384-dim `vector` column (pgvector,
matching fastembed's `bge-small-en-v1.5`) populated later by #21.

Requires the `vector` extension, enabled here with `CREATE EXTENSION IF
NOT EXISTS` -- the `pgvector/pgvector:pg16` image ships the extension,
but each database must still enable it once before `vector(384)` columns
can be created.
"""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "6e4f17b2d041"
down_revision: str | None = "f0a9e574ee03"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "source",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("watch_id", sa.Uuid(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "discovered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("entities", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(384), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("triage_note", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('discovered', 'extracted', 'extract_failed', "
            "'triaged_out', 'processed')",
            name="ck_source_status",
        ),
        sa.ForeignKeyConstraint(["watch_id"], ["watch.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("watch_id", "url", name="uq_source_watch_id_url"),
    )


def downgrade() -> None:
    """Downgrade schema.

    Drops the `source` table only -- the `vector` extension is left in
    place since later tables (later issues) also need it.
    """
    op.drop_table("source")
