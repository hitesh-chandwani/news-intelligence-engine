"""FastAPI routers, one module per resource (`design.md` §15).

`watch.py` (#34) is the first. `context.py`/`preferences.py`/`events.py`/
`notifications.py`/`pipeline.py` land in later tasks per `design.md` §12's
full endpoint table -- not built out here (issue #34's "Out of scope").
"""

from __future__ import annotations
