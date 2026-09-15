"""FastAPI dependencies shared across `nie.web` routers (issue #34).

Split out from `nie.web.app` (one of the two locations the issue allows)
so routers can import `get_session` without creating a circular import
with `app.py` (which imports the routers to include them).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from nie.db import async_session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an `AsyncSession` from `nie.db.async_session_factory`.

    A test overrides this via `app.dependency_overrides[get_session]` with
    its own session factory built from `nie.db.create_engine`/
    `create_session_factory` (same pattern as `tests/test_run.py`), rather
    than this module-level singleton -- see `tests/test_web_watch.py`.
    """
    async with async_session_factory() as session:
        yield session
