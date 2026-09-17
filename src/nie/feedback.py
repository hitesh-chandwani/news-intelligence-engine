"""Feedback data layer (#39; `design.md` §4, §11, §12, FR-025, FR-026).

Write-only for the HTTP layer's `POST /events/{id}/feedback` (added in
`nie.web.routers.events`): a single `submit_feedback` function that
validates and inserts one `nie.models.Feedback` row, mirroring the shape
of `nie.notifications.inbox.mark_read` (#33) -- HTTP-agnostic, raises a
plain `ValueError` subclass on each of its two failure modes rather than
an `HTTPException`, and commits itself before returning since there is no
wrapping "runner" here, same reasoning `mark_read`'s docstring gives.

`withdraw_feedback` (#51) is the hard-delete counterpart, backing `DELETE
/events/{event_id}/feedback/{feedback_id}` (added in
`nie.web.routers.events`): same HTTP-agnostic shape, same "raise a
`ValueError` subclass, commit itself" contract as `submit_feedback`.
Editing a feedback row in place is explicitly out of scope (#58).

Reading feedback back out (the `FeedbackBucket` context-bundle read
#27/#28's `src/nie/pipeline/score.py` already implements) is out of scope
here -- this module only produces/removes the rows that read side
consumes; `FeedbackBucket` needs no code change for a withdrawn row to
stop counting since it queries live table state.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from nie.models import Event, Feedback

# The exact value set `nie.models.Feedback.verdict`'s `CheckConstraint`
# allows -- kept here rather than re-derived from the DB so `submit_feedback`
# can reject an unrecognized value before ever touching the database (the
# issue's Constraints: "verdict must be validated ... before insert, not
# left to the DB constraint to raise an unhandled IntegrityError/500").
_VERDICT_VALUES = {
    "useful",
    "not_useful",
    "too_many_similar",
    "more_like_this",
    "less_of_this",
}


class UnknownVerdictError(ValueError):
    """Raised by `submit_feedback` when `verdict` is not one of the 5
    values `_VERDICT_VALUES` (== `Feedback.verdict`'s `CheckConstraint`)
    allows. A `ValueError` subclass -- same "raise ValueError, let the
    router translate it" shape `mark_read` (#33) established -- so the
    router can distinguish this failure mode (-> `422`) from
    `EventNotFoundError` (-> `404`) with a plain `except` clause per
    exception type, while both still satisfy "raise a ValueError" as
    ordinary `ValueError`s to any other caller.
    """


class EventNotFoundError(ValueError):
    """Raised by `submit_feedback` when no `Event` row exists with the
    given `event_id`. A `ValueError` subclass, same reasoning as
    `UnknownVerdictError` above.
    """


class FeedbackNotFoundError(ValueError):
    """Raised by `withdraw_feedback` (#51) when no `Feedback` row exists
    with the given `feedback_id`, or when one does but its `event_id`
    doesn't match the `event_id` passed in -- both collapse to the same
    `404` in the router, same "don't leak whether a mismatched id exists
    under a different event" reasoning `_get_context_item_or_404`
    (`context.py`) and `_get_event_or_404` (`events.py`) already follow
    for their own not-found cases. A `ValueError` subclass, same shape as
    `UnknownVerdictError`/`EventNotFoundError` above.
    """


async def submit_feedback(
    session: AsyncSession, event_id: uuid.UUID, verdict: str, note: str | None = None
) -> Feedback:
    """Validate `verdict`, look up `event_id`, and insert one `Feedback`
    row.

    Verdict is checked first (no DB round-trip needed to reject it) via
    `_VERDICT_VALUES`, raising `UnknownVerdictError` on a mismatch. The
    `Event` row is then loaded by `event_id`, raising `EventNotFoundError`
    when it doesn't exist. `Feedback.watch_id` is taken from the loaded
    event's own `watch_id` -- correct in general (a feedback row belongs
    to whichever watch surfaced the event it's about), and in this
    single-watch MVP that watch is always the Silver watch, matching the
    Silver-watch resolution every router in `nie.web.routers` already
    performs (see the issue's Constraints) without a second, separate
    lookup-by-slug here.

    `note` (#50) is optional free text: `None` or blank/whitespace-only is
    normalized to `None` before insert (an empty textarea must not persist
    `note=""`); a non-blank `note` is stored with leading/trailing
    whitespace stripped, otherwise verbatim -- no length cap, `note` is an
    unconstrained `Text` column (`design.md` §4).

    Commits itself before returning (there is no wrapping runner here,
    unlike the pipeline stage functions) and refreshes the row afterward
    so the caller sees the DB-assigned `created_at`.
    """
    if verdict not in _VERDICT_VALUES:
        raise UnknownVerdictError(
            f"Unrecognized feedback verdict: {verdict!r}; "
            f"must be one of {sorted(_VERDICT_VALUES)}"
        )

    event = await session.get(Event, event_id)
    if event is None:
        raise EventNotFoundError(f"No event found with id {event_id!r}")

    normalized_note = note.strip() if note is not None and note.strip() else None
    feedback = Feedback(
        watch_id=event.watch_id, event_id=event.id, verdict=verdict, note=normalized_note
    )
    session.add(feedback)
    await session.commit()
    await session.refresh(feedback)
    return feedback


async def withdraw_feedback(
    session: AsyncSession, event_id: uuid.UUID, feedback_id: uuid.UUID
) -> None:
    """Hard-delete one `Feedback` row (#51), for "Undo" on a
    just-submitted confirmation.

    Looks up `feedback_id` directly (a `Feedback.id` primary-key lookup,
    same `session.get` shape `submit_feedback` uses for `Event`), raising
    `FeedbackNotFoundError` when no such row exists at all, or when it
    exists but its `event_id` doesn't match the `event_id` given here --
    a mismatched id is treated identically to a nonexistent one rather
    than a separate error, same reasoning `FeedbackNotFoundError`'s
    docstring gives.

    Hard delete only (no soft-delete/`updated_at` column on `feedback` --
    `design.md` §4's schema for this table is unchanged by #51): the row
    is deleted and the delete is committed before returning, same
    "commits itself, no wrapping runner" contract `submit_feedback`
    documents. A second call against the same `feedback_id` (double-
    withdraw) finds nothing and raises `FeedbackNotFoundError` again, not
    a silent no-op success.
    """
    feedback = await session.get(Feedback, feedback_id)
    if feedback is None or feedback.event_id != event_id:
        raise FeedbackNotFoundError(
            f"No feedback found with id {feedback_id!r} for event {event_id!r}"
        )

    await session.delete(feedback)
    await session.commit()
