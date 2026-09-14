"""Notification channels (#31, #32).

Each channel module (currently just `email.py`) exposes a pure
`render_*` function plus a `send_*` coroutine that reads `Settings` and
talks to the channel's external API. Nothing in this package is wired
into the pipeline yet -- see `email.py`'s module docstring for the
"helper first, wiring later" precedent this follows (#48 wires channel
sends into the notify stage).
"""

from __future__ import annotations
