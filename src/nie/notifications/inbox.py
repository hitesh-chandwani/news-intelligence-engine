"""In-app notification inbox data layer (#33).

Read/query access only for the future inbox UI (#38): listing a watch's
notifications newest-first (each row already carries its read/unread
state via `Notification.read_at`) and marking one read. Creating the
`Notification` row itself (the `inapp` "send") is #48's job via a plain
`session.add(Notification(...))` inside `notify_stage` -- see #48's own
acceptance criteria and the issue's Goal section -- so no send-side
function belongs here.

Unlike the pipeline stage functions in `src/nie/pipeline/*.py`, this
module has no wrapping "runner" that commits after it runs. `list_notifications`
performs no writes and calls no `commit`. `mark_read` is the write path:
since #38's HTTP route will call it directly and needs the write
durable, it calls `await session.commit()` itself before returning.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import Notification


async def list_notifications(session: AsyncSession, watch_id: uuid.UUID) -> list[Notification]:
    """Return every `Notification` row for `watch_id`, newest first.

    No pagination/`limit`/`offset` and no `unread_only` filter -- see the
    issue's Goal/acceptance criteria. Returns `[]` when `watch_id` has no
    notifications, or does not exist as a `Watch` row at all -- this
    function only queries `notification`, it does not validate `watch_id`
    against `watch`. Performs no writes and calls no `commit`.
    """
    result = await session.execute(
        select(Notification)
        .where(Notification.watch_id == watch_id)
        .order_by(Notification.created_at.desc())
    )
    return list(result.scalars().all())


async def mark_read(session: AsyncSession, notification_id: uuid.UUID) -> Notification:
    """Mark one `Notification` read, set-once-idempotent.

    Loads the row by `id`, sets `read_at` to the current UTC,
    timezone-aware time if it is currently `None`, and returns the row.
    Calling this a second time on an already-read notification is a
    no-op -- `read_at` keeps its original value. Commits itself before
    returning (there is no wrapping runner here, unlike the pipeline
    stage functions). Raises `ValueError` -- mirroring the
    `raise ValueError(f"Unrecognized discovery provider: {name!r}")`
    style in `src/nie/pipeline/discover.py` -- when no `Notification` row
    with that `id` exists; no commit happens on that path.
    """
    notification = await session.get(Notification, notification_id)
    if notification is None:
        raise ValueError(f"No notification found with id {notification_id!r}")

    if notification.read_at is None:
        notification.read_at = datetime.now(UTC)
        await session.commit()

    return notification
