"""Seed data and idempotent seed functions.

Per ``_docs/design.md`` §15, each seedable table gets its own module here
(e.g. ``categories.py`` for #7). #14's seed script imports and calls these
functions; this package owns zero wiring/CLI logic itself.
"""

from __future__ import annotations
