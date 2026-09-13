"""The discover->notify pipeline (`design.md` §5).

`runner.py` (#18) is the only module here for now -- each real stage
(discover, extract, triage, ...) lands in its own module starting with
#19, swapped into `runner.STAGE_REGISTRY` one at a time.
"""

from __future__ import annotations
