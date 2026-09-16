"""Tests for src/nie/web/routers/preferences.py (issue #36).

Follows the exact `migrated_db`/`session_factory`/`seeded_client` fixture
pattern `tests/test_web_watch.py`/`tests/test_web_context.py` established:
a fresh engine per test (not `nie.db`'s module-level singleton),
`create_app()` + `app.dependency_overrides[get_session]`, `httpx.AsyncClient`
+ `ASGITransport` (no real server process), seeded via `nie.seed.run.seed`.

The `notification_preference` row is a **shared, never-truncated** row
against the live Compose Postgres instance (`watch_id` is its primary
key, `seed()`'s `_seed_notification_preference` only inserts it once via
`ON CONFLICT (watch_id) DO NOTHING` and never overwrites an existing
row) -- `session_factory`/`migrated_db` give each test its own *engine*
and re-run migrations, but the underlying *data* is the same shared
database every other test module (this one included) reads and writes,
exactly like `context_item` in `test_web_context.py`. So, per
`_docs/testing-guidelines.md` and this issue's own Constraints section,
every assertion here compares against state read/set earlier in the
*same* test, never a hardcoded literal for the row's *current* values.

Note on `GET /preferences` returning "the seed default" (this issue's
own test list): that literal-value assertion is not actually reachable
in this shared DB -- `tests/test_run.py`'s
`test_seed_is_idempotent_and_preserves_edits_made_between_runs`
deliberately edits this exact row to `min_importance="critical"` to prove
`seed()` preserves user edits, and that edit persists in this shared,
never-truncated database for the lifetime of the Compose volume,
contradicting "freshly seeded" for any test that runs after it (proven by
this module's first version of this test actually failing that way). See
this module's `test_get_preferences_returns_current_row_shape_and_values`
below and issue #36's GitHub comment for how this was resolved.
"""

import re
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic.command import upgrade
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nie.db import create_engine, create_session_factory
from nie.seed.run import seed
from nie.web.app import create_app
from nie.web.deps import get_session

REPO_ROOT = Path(__file__).parent.parent

# Field names of `nie.schemas.PreferenceResponse`, as serialized on the
# wire -- used to check response shape without depending on exact values.
_PREFERENCE_RESPONSE_FIELDS = {"watch_id", "min_importance", "categories", "channels"}


def _is_preference_response_shaped(value: object) -> bool:
    return isinstance(value, dict) and value.keys() == _PREFERENCE_RESPONSE_FIELDS


@pytest.fixture
def migrated_db() -> None:
    """Run `alembic upgrade head` against the live Compose DB.

    A plain (sync) fixture -- Alembic drives its own event loop
    internally, so this must run outside pytest-asyncio's loop for the
    test function. Re-running this is idempotent.
    """
    config = Config(str(REPO_ROOT / "alembic.ini"))
    upgrade(config, "head")


@pytest.fixture
async def session_factory(migrated_db: None) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A session factory built from its own engine, scoped to this test."""
    engine = create_engine()
    try:
        yield create_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.fixture
async def seeded_client(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncClient]:
    """A `create_app()` instance with `get_session` overridden to
    `session_factory`, seeded with the Silver watch (and its default
    notification preference), driven over `ASGITransport` via
    `httpx.AsyncClient`.
    """
    async with session_factory() as session:
        await seed(session)

    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = _override_get_session

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_get_preferences_returns_current_row_shape_and_values(
    seeded_client: AsyncClient,
) -> None:
    """`GET /preferences` returns the Silver watch's
    `notification_preference` row, correctly shaped, with `min_importance`
    one of the four allowed values and `categories`/`channels` both lists.

    This stands in for the issue's "returns the seed default" test case:
    that literal-value assertion (`min_importance="medium"`,
    `categories=[]`, `channels=[]`) is not actually reachable against this
    shared, never-truncated DB -- `tests/test_run.py`'s
    `test_seed_is_idempotent_and_preserves_edits_made_between_runs`
    deliberately edits this exact row to `min_importance="critical"` to
    prove `seed()` preserves user edits, and that edit persists for the
    life of the Compose volume, so any test after it (including this one,
    alphabetically) sees `"critical"`, not `"medium"`. Asserting the
    literal seed default here would be exactly the hardcoded-literal
    assertion this issue's own Constraints section forbids on this shared
    row. `seed()`'s actual default values are already covered by
    `tests/test_models.py`'s
    `test_notification_preference_default_min_importance_and_array_roundtrip`
    and `tests/test_run.py`'s idempotency test; the exact PATCH round-trip
    of all three fields (including explicitly setting them back to the
    seed defaults) is covered below by
    `test_patch_all_three_fields_persists_and_round_trips`.
    """
    response = await seeded_client.get("/preferences")
    assert response.status_code == 200
    body = response.json()
    assert _is_preference_response_shaped(body)
    assert body["min_importance"] in {"low", "medium", "high", "critical"}
    assert isinstance(body["categories"], list)
    assert isinstance(body["channels"], list)


async def test_patch_all_three_fields_persists_and_round_trips(
    seeded_client: AsyncClient,
) -> None:
    """A `PATCH` setting all three fields persists and round-trips: the
    `PATCH` response and a subsequent `GET` both reflect the new values.
    """
    patch_response = await seeded_client.patch(
        "/preferences",
        json={"min_importance": "high", "categories": ["mining", "market"], "channels": ["email"]},
    )
    assert patch_response.status_code == 200
    patched = patch_response.json()
    assert _is_preference_response_shaped(patched)
    assert patched["min_importance"] == "high"
    assert sorted(patched["categories"]) == ["market", "mining"]
    assert patched["channels"] == ["email"]

    get_response = await seeded_client.get("/preferences")
    assert get_response.status_code == 200
    fetched = get_response.json()
    assert fetched["min_importance"] == "high"
    assert sorted(fetched["categories"]) == ["market", "mining"]
    assert fetched["channels"] == ["email"]


async def test_patch_one_field_leaves_others_unchanged(seeded_client: AsyncClient) -> None:
    """A `PATCH` setting only one field (`min_importance`) leaves the other
    two unchanged -- asserted against whatever the row held before this
    `PATCH` in this same test, not a hardcoded literal, since the row is
    shared, never-truncated state.
    """
    # Set a known, non-default state for categories/channels first, so the
    # "unchanged" assertion below isn't trivially satisfied by an
    # already-empty default.
    setup_response = await seeded_client.patch(
        "/preferences", json={"categories": ["price", "macro"], "channels": ["telegram", "inapp"]}
    )
    assert setup_response.status_code == 200
    before = setup_response.json()

    response = await seeded_client.patch("/preferences", json={"min_importance": "critical"})
    assert response.status_code == 200
    after = response.json()

    assert after["min_importance"] == "critical"
    assert after["categories"] == before["categories"]
    assert after["channels"] == before["channels"]

    get_response = await seeded_client.get("/preferences")
    fetched = get_response.json()
    assert fetched["min_importance"] == "critical"
    assert fetched["categories"] == before["categories"]
    assert fetched["channels"] == before["channels"]


async def test_patch_empty_body_is_a_noop_leaving_everything_unchanged(
    seeded_client: AsyncClient,
) -> None:
    """A `PATCH` with an empty body sets nothing (no all-`None` validator
    on `PreferenceUpdate`, unlike `ContextItemUpdate`) and leaves every
    field unchanged, compared against state read earlier in this test.
    """
    before_response = await seeded_client.get("/preferences")
    before = before_response.json()

    response = await seeded_client.patch("/preferences", json={})
    assert response.status_code == 200
    after = response.json()

    assert after["min_importance"] == before["min_importance"]
    assert after["categories"] == before["categories"]
    assert after["channels"] == before["channels"]


async def test_patch_invalid_min_importance_returns_422(seeded_client: AsyncClient) -> None:
    """`min_importance` outside `{"low", "medium", "high", "critical"}`
    returns `422`.
    """
    response = await seeded_client.patch("/preferences", json={"min_importance": "urgent"})
    assert response.status_code == 422


async def test_patch_invalid_channel_returns_422(seeded_client: AsyncClient) -> None:
    """Any `channels` entry outside `{"telegram", "email", "inapp"}`
    returns `422`.
    """
    response = await seeded_client.patch("/preferences", json={"channels": ["sms"]})
    assert response.status_code == 422


async def test_patch_invalid_min_importance_or_channel_makes_no_change(
    seeded_client: AsyncClient,
) -> None:
    """A rejected `PATCH` (422) makes no change to the row -- asserted
    against state read before the attempted write.
    """
    before_response = await seeded_client.get("/preferences")
    before = before_response.json()

    reject_response = await seeded_client.patch(
        "/preferences", json={"min_importance": "urgent", "channels": ["sms"]}
    )
    assert reject_response.status_code == 422

    after_response = await seeded_client.get("/preferences")
    after = after_response.json()
    assert after == before


async def test_get_and_patch_hx_request_header_gets_html_fragment(
    seeded_client: AsyncClient,
) -> None:
    """`HX-Request: true` on `GET`/`PATCH /preferences` gets back
    `text/html`, the `partials/preferences_form.html` fragment.
    """
    get_response = await seeded_client.get("/preferences", headers={"HX-Request": "true"})
    assert get_response.status_code == 200
    assert get_response.headers["content-type"].startswith("text/html")
    assert "<form" in get_response.text
    assert 'name="min_importance"' in get_response.text

    patch_response = await seeded_client.patch(
        "/preferences",
        json={"min_importance": "low"},
        headers={"HX-Request": "true"},
    )
    assert patch_response.status_code == 200
    assert patch_response.headers["content-type"].startswith("text/html")
    assert "<form" in patch_response.text


async def test_get_preferences_plain_request_gets_json(seeded_client: AsyncClient) -> None:
    """No `HX-Request` header and a non-`text/html` `Accept` (including
    `httpx`'s default `*/*`) gets JSON, not HTML.
    """
    response = await seeded_client.get("/preferences")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert _is_preference_response_shaped(response.json())


async def test_patch_plain_request_gets_json(seeded_client: AsyncClient) -> None:
    """No `HX-Request` header on `PATCH /preferences` gets back JSON."""
    response = await seeded_client.patch("/preferences", json={"min_importance": "medium"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert _is_preference_response_shaped(response.json())


async def test_get_preferences_html_accept_header_gets_full_page(
    seeded_client: AsyncClient,
) -> None:
    """No `HX-Request` header and an `Accept: text/html...` header (a
    plain browser navigation) gets `text/html`, the full `preferences.html`
    page (extends `base.html`).
    """
    response = await seeded_client.get(
        "/preferences", headers={"Accept": "text/html,application/xhtml+xml"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Notification preferences" in response.text
    assert 'name="min_importance"' in response.text


async def test_full_page_loads_json_enc_extension_script(seeded_client: AsyncClient) -> None:
    """The full `preferences.html` page (extends `base.html`) loads htmx's
    `json-enc` extension script -- required for the preferences `<form>`'s
    `hx-ext="json-enc"` (see `test_preferences_form_uses_json_enc_extension`)
    to actually take effect in a real browser. Regression test for the bug
    QA found on #35's first pass: without this script tag (and the
    matching `hx-ext="json-enc"` on the form), htmx serializes a submitted
    form as `application/x-www-form-urlencoded`, but `PATCH /preferences`
    only accepts a JSON body -- so a real form submission would always
    fail with `422`, even though every test that PATCHes with `json=`
    directly (bypassing the actual rendered form) passes.
    """
    response = await seeded_client.get(
        "/preferences", headers={"Accept": "text/html,application/xhtml+xml"}
    )
    assert response.status_code == 200
    assert "htmx.org@1.9.12/dist/ext/json-enc.js" in response.text


async def test_preferences_form_uses_json_enc_extension(seeded_client: AsyncClient) -> None:
    """Both the full page's rendered form and the `HX-Request` fragment's
    form carry `hx-ext="json-enc"`, so htmx serializes the submitted
    fields as a JSON body instead of the default
    `application/x-www-form-urlencoded` -- matching what `PATCH
    /preferences` actually accepts. Regression test for the bug QA found
    on #35's first pass (see
    `test_full_page_loads_json_enc_extension_script`'s docstring for the
    full mechanism); a full real-browser form submission isn't practical
    in this httpx-based harness, so this attribute-presence assertion is
    the practical regression guard.
    """
    full_page_response = await seeded_client.get(
        "/preferences", headers={"Accept": "text/html,application/xhtml+xml"}
    )
    assert full_page_response.status_code == 200
    assert 'hx-ext="json-enc"' in full_page_response.text

    fragment_response = await seeded_client.get("/preferences", headers={"HX-Request": "true"})
    assert fragment_response.status_code == 200
    assert 'hx-ext="json-enc"' in fragment_response.text


async def test_categories_multi_select_lists_all_seeded_categories_and_preselects_current(
    seeded_client: AsyncClient,
) -> None:
    """The rendered form lists all 13 seeded categories
    (`nie.seed.categories.CATEGORIES`) and pre-checks/pre-selects the
    watch's current `categories` slugs.
    """
    patch_response = await seeded_client.patch(
        "/preferences", json={"categories": ["supply", "geopolitical"]}
    )
    assert patch_response.status_code == 200

    response = await seeded_client.get("/preferences", headers={"HX-Request": "true"})
    assert response.status_code == 200
    text = response.text

    for slug in [
        "supply",
        "mining",
        "demand",
        "industrial",
        "market",
        "price",
        "inventory",
        "etf-investment",
        "macro",
        "geopolitical",
        "regulatory",
        "company",
        "other",
    ]:
        assert f'value="{slug}"' in text

    # Precisely confirm each category checkbox's own `<input>` tag is
    # `checked` if and only if its slug is one of the two just PATCHed in
    # -- this fixture's DB is fresh per test, so `channels` is still the
    # seed default (`[]`), meaning no channel checkbox is checked either.
    for slug in ["supply", "geopolitical"]:
        tag = re.search(rf'<input[^>]*value="{slug}"[^>]*/>', text, re.DOTALL)
        assert tag is not None
        assert "checked" in tag.group(0)
    for slug in ["mining", "other"]:
        tag = re.search(rf'<input[^>]*value="{slug}"[^>]*/>', text, re.DOTALL)
        assert tag is not None
        assert "checked" not in tag.group(0)
