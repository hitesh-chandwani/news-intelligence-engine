"""Async SQLAlchemy engine + session factory.

Both are built from ``Settings().database_url`` — never read
``os.environ`` directly (see ``_docs/design.md`` §14 and
``nie.config.Settings``). This module owns zero application/business
logic; it only wires up the engine and session factory that the rest of
the app (and Alembic, via ``alembic/env.py``) build on.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nie.config import Settings


def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine from ``Settings().database_url``."""
    settings = settings or Settings()
    return create_async_engine(settings.database_url)


def create_session_factory(
    engine: AsyncEngine | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Create an ``async_sessionmaker`` bound to the given (or a new) engine."""
    engine = engine or create_engine()
    return async_sessionmaker(engine, expire_on_commit=False)


# Module-level default engine + session factory, built from the process
# `Settings()` at import time — the common case for the app and tests.
engine: AsyncEngine = create_engine()
async_session_factory: async_sessionmaker[AsyncSession] = create_session_factory(engine)
