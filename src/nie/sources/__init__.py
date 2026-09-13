"""Discovery providers (`_docs/design.md` §7).

Each provider (`StubProvider` here, `RssProvider` in #17) implements the
`DiscoveryProvider` Protocol defined in `base.py`. This package owns no
wiring/selection logic itself -- turning `DISCOVERY_PROVIDERS` config into
instantiated providers is #18.
"""

from __future__ import annotations
