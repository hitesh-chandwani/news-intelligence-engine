"""Notification gate + payload builder (#30, `design.md` §5 stage 10,
§10, §8 FR-021, FR-022).

Neither `notify_gate` nor `build_notification_payload` is wired into
`STAGE_REGISTRY` or called from anywhere yet -- both are pure/DB-read-only
helper functions shipped ahead of, and independent from, their eventual
call site (#48), same "helper first, wiring later" precedent #22/#27 set.

`notify_gate` is synchronous with no DB access: it takes `reason` as an
explicit caller-supplied parameter rather than inferring "new-or-material"
itself -- see the issue's "Resolving 'how does the gate know
new-or-material'" section for why that inference deliberately does not
live here (filed as #48).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from nie.models import Event, NotificationPreference

_IMPORTANCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def notify_gate(
    event: Event,
    reason: Literal["new-event", "material-update"] | None,
    category_slugs: Sequence[str],
    preference: NotificationPreference,
) -> bool:
    """Decide whether `event` should generate a notification for the watch
    `preference` belongs to, per `design.md` §5 stage 10. Pure, synchronous,
    no `session` parameter -- operates only on already-loaded values.

    All four conditions must hold:

    1. `reason is not None` -- the caller determined this run either
       created `event` (`"new-event"`) or materially updated it
       (`"material-update"`); `None` means neither happened this run, so
       there is nothing to notify about.
    2. `event.relevance not in (None, "irrelevant")` -- `None` means not
       yet scored (fail closed, don't notify, don't raise); `"irrelevant"`
       is an explicit scored verdict to suppress.
    3. `event.importance is not None` and ranks `>= preference.min_importance`
       on `_IMPORTANCE_ORDER` -- `None` means not yet scored (fail closed).
    4. `preference.categories` is empty (the documented "empty = all
       categories" convention), or shares at least one slug with
       `category_slugs`.

    `category_slugs` is supplied by the caller (loaded the same way
    `build_notification_payload` loads `categories`) rather than queried
    here, keeping this function DB-free and trivially unit-testable.
    """
    if reason is None:
        return False

    if event.relevance is None or event.relevance == "irrelevant":
        return False

    if event.importance is None:
        return False
    if _IMPORTANCE_ORDER[event.importance] < _IMPORTANCE_ORDER[preference.min_importance]:
        return False

    if preference.categories and not (set(preference.categories) & set(category_slugs)):
        return False

    return True
