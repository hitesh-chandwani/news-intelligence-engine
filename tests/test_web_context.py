"""Tests for src/nie/web/routers/context.py (issue #35).

Follows the exact `migrated_db`/`session_factory`/`seeded_client` fixture
pattern `tests/test_web_watch.py` established for issue #34: a fresh
engine per test (not `nie.db`'s module-level singleton), `create_app()` +
`app.dependency_overrides[get_session]`, `httpx.AsyncClient` +
`ASGITransport` (no real server process), seeded via `nie.seed.run.seed`.

This repo's test DB is a shared, never-truncated Compose Postgres
instance, so this module follows the two pitfalls `test_web_watch.py`'s
own QA fix already documents -- and issue #35's own test-approach section
repeats them explicitly:

- **Never an exact-count assertion.** Other test modules (and other runs
  of this one) may have already left `context_item` rows in this DB.
  Every `GET /context` assertion here is containment-based ("the item I
  created is present", "at least one system item is present"), never
  "the list has exactly N items".
- **Never delete a row to manufacture a 403/404 fixture.** `context_item`
  rows may have FK dependents, and Postgres enforces FKs at `DELETE` time
  (not commit time), so even an uncommitted, later-rolled-back delete
  fails before rollback ever comes into play (`test_web_watch.py`'s
  `test_missing_silver_watch_returns_404` docstring covers the identical
  mechanism for `watch`). The 403 tests fetch a *real* `kind="system"` id
  via `GET /context` (seeded by `seed()` from `silver_context.md`) and
  only *attempt* the forbidden write, re-`GET`ing afterwards to confirm
  it's unchanged. The 404 tests use a fresh `uuid.uuid4()` that provably
  matches no row, never a deleted one.

The create->update->delete round-trip test creates its own fresh row (a
`uuid4()`-embedded label so it can't collide with anything already in the
shared DB) and is the one test permitted to delete a `context_item` row --
it's a `kind="user"` row this test itself created, with no FK dependents.
"""

import uuid
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

# Field names of `nie.schemas.ContextItemResponse`, as serialized on the
# wire -- used to check response shape without depending on exact values
# other than the ones each test controls.
_CONTEXT_ITEM_RESPONSE_FIELDS = {"id", "kind", "label", "body", "created_at", "updated_at"}


def _is_context_item_response_shaped(value: object) -> bool:
    return isinstance(value, dict) and value.keys() == _CONTEXT_ITEM_RESPONSE_FIELDS


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
    `session_factory`, seeded with the Silver watch (and its system
    context), driven over `ASGITransport` via `httpx.AsyncClient`.
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


async def test_create_update_delete_round_trip(seeded_client: AsyncClient) -> None:
    """`POST /context` -> `PATCH /context/{id}` -> `DELETE /context/{id}`
    -> `GET /context` no longer includes the id, on a fresh row this test
    creates itself (unique label/body via `uuid4()`, so it can't collide
    with rows other tests/runs left behind in this shared DB).
    """
    unique = uuid.uuid4()
    create_response = await seeded_client.post(
        "/context",
        json={"label": f"Test label {unique}", "body": f"Test body {unique}"},
    )
    assert create_response.status_code == 201
    created = create_response.json()
    assert _is_context_item_response_shaped(created)
    assert created["kind"] == "user"
    assert created["label"] == f"Test label {unique}"
    assert created["body"] == f"Test body {unique}"
    item_id = created["id"]

    update_response = await seeded_client.patch(
        f"/context/{item_id}", json={"label": f"Updated label {unique}"}
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert _is_context_item_response_shaped(updated)
    assert updated["id"] == item_id
    assert updated["label"] == f"Updated label {unique}"
    # body was not sent in the PATCH, so it must be unchanged.
    assert updated["body"] == f"Test body {unique}"

    delete_response = await seeded_client.delete(f"/context/{item_id}")
    assert delete_response.status_code == 204
    assert delete_response.content == b""

    list_response = await seeded_client.get("/context")
    assert list_response.status_code == 200
    ids = {item["id"] for item in list_response.json()}
    assert item_id not in ids


async def test_get_context_contains_created_item_and_a_system_item(
    seeded_client: AsyncClient,
) -> None:
    """`GET /context` (JSON) is containment-based: the item this test
    created is present, and at least one `kind="system"` item (seeded
    from `silver_context.md`) is present -- never an exact-length
    assertion, since other test modules may have already created rows in
    this shared DB.
    """
    unique = uuid.uuid4()
    create_response = await seeded_client.post(
        "/context",
        json={"label": f"Containment label {unique}", "body": f"Containment body {unique}"},
    )
    assert create_response.status_code == 201
    created_id = create_response.json()["id"]

    list_response = await seeded_client.get("/context")
    assert list_response.status_code == 200
    items = list_response.json()
    assert all(_is_context_item_response_shaped(item) for item in items)

    ids = {item["id"] for item in items}
    assert created_id in ids

    kinds = {item["kind"] for item in items}
    assert "system" in kinds
    assert "user" in kinds


async def _get_a_system_item_id(client: AsyncClient) -> str:
    """Fetch a real `kind="system"` id via `GET /context` -- seeded by
    `seed()` from `silver_context.md`, so at least one always exists.
    Never deletes or otherwise modifies any row.
    """
    response = await client.get("/context")
    assert response.status_code == 200
    system_items = [item for item in response.json() if item["kind"] == "system"]
    assert system_items, "expected at least one seeded kind='system' context item"
    return str(system_items[0]["id"])


async def test_patch_on_system_item_returns_403_and_makes_no_change(
    seeded_client: AsyncClient,
) -> None:
    """`PATCH /context/{id}` on a real, existing `kind="system"` row
    returns `403` (not `404` -- the row exists and was just visible via
    `GET /context`) and leaves the row unchanged. Only *attempts* the
    forbidden write; never deletes a system row to set the test up.
    """
    system_id = await _get_a_system_item_id(seeded_client)

    before_response = await seeded_client.get("/context")
    before_item = next(item for item in before_response.json() if item["id"] == system_id)

    response = await seeded_client.patch(
        f"/context/{system_id}", json={"label": "attempted takeover"}
    )
    assert response.status_code == 403
    assert "detail" in response.json()

    after_response = await seeded_client.get("/context")
    after_item = next(item for item in after_response.json() if item["id"] == system_id)
    assert after_item == before_item


async def test_delete_on_system_item_returns_403_and_makes_no_change(
    seeded_client: AsyncClient,
) -> None:
    """`DELETE /context/{id}` on a real, existing `kind="system"` row
    returns `403` and leaves it present. Only *attempts* the forbidden
    delete; never actually deletes the system row.
    """
    system_id = await _get_a_system_item_id(seeded_client)

    response = await seeded_client.delete(f"/context/{system_id}")
    assert response.status_code == 403
    assert "detail" in response.json()

    after_response = await seeded_client.get("/context")
    ids = {item["id"] for item in after_response.json()}
    assert system_id in ids


async def test_patch_on_nonexistent_id_returns_404(seeded_client: AsyncClient) -> None:
    """`PATCH /context/{id}` on a fresh `uuid4()` that provably matches no
    row (never a deleted one -- `context_item` rows may have FK
    dependents, and an uncommitted delete still fails at `DELETE` time in
    Postgres) returns `404`.
    """
    missing_id = uuid.uuid4()
    response = await seeded_client.patch(f"/context/{missing_id}", json={"label": "x"})
    assert response.status_code == 404
    assert "detail" in response.json()


async def test_delete_on_nonexistent_id_returns_404(seeded_client: AsyncClient) -> None:
    """`DELETE /context/{id}` on a fresh `uuid4()` returns `404`."""
    missing_id = uuid.uuid4()
    response = await seeded_client.delete(f"/context/{missing_id}")
    assert response.status_code == 404
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "payload",
    [
        {"label": "", "body": "non-blank body"},
        {"label": "   ", "body": "non-blank body"},
        {"label": "non-blank label", "body": ""},
        {"body": "missing label"},
        {"label": "missing body"},
    ],
)
async def test_create_with_blank_or_missing_field_returns_422(
    seeded_client: AsyncClient, payload: dict[str, str]
) -> None:
    """`POST /context` with a missing or blank `label`/`body` returns
    `422`.
    """
    response = await seeded_client.post("/context", json=payload)
    assert response.status_code == 422


async def test_update_with_neither_field_set_returns_422(seeded_client: AsyncClient) -> None:
    """`PATCH /context/{id}` with neither `label` nor `body` set returns
    `422` -- the model validator on `ContextItemUpdate` rejects it before
    the router is even reached, so any id (real or not) exercises this.
    """
    system_id = await _get_a_system_item_id(seeded_client)
    response = await seeded_client.patch(f"/context/{system_id}", json={})
    assert response.status_code == 422


async def test_get_context_hx_request_header_gets_html_fragment(
    seeded_client: AsyncClient,
) -> None:
    """`HX-Request: true` on `GET /context` gets `text/html`, the
    `partials/context_lists.html` fragment.
    """
    response = await seeded_client.get("/context", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "System context" in response.text
    assert "Your context" in response.text


async def test_get_context_html_accept_header_gets_full_page(
    seeded_client: AsyncClient,
) -> None:
    """No `HX-Request` header and an `Accept: text/html...` header (a
    plain browser navigation) gets `text/html`, the full `context.html`
    page (extends `base.html`).
    """
    response = await seeded_client.get(
        "/context", headers={"Accept": "text/html,application/xhtml+xml"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Background context" in response.text
    assert "System context" in response.text


async def test_full_page_loads_json_enc_extension_script(seeded_client: AsyncClient) -> None:
    """The full `context.html` page (extends `base.html`) loads htmx's
    `json-enc` extension script -- required for the add-item/edit `<form>`
    elements' `hx-ext="json-enc"` (see
    `test_user_items_forms_use_json_enc_extension`) to actually take
    effect in a real browser. Regression test for the bug QA found on
    #35's first pass: without this script tag (and the matching
    `hx-ext="json-enc"` on the forms), htmx serializes a submitted form as
    `application/x-www-form-urlencoded`, but `POST /context`/`PATCH
    /context/{id}` only accept a JSON body -- so a real form submission
    always failed with `422`, even though every test that posted `json=`
    directly (bypassing the actual rendered form) passed.
    """
    response = await seeded_client.get(
        "/context", headers={"Accept": "text/html,application/xhtml+xml"}
    )
    assert response.status_code == 200
    assert "htmx.org@1.9.12/dist/ext/json-enc.js" in response.text


async def test_user_items_forms_use_json_enc_extension(seeded_client: AsyncClient) -> None:
    """Both the add-item form and a user item's edit form in the rendered
    `partials/context_user_items.html` fragment carry `hx-ext="json-enc"`,
    so htmx serializes their submitted fields as a JSON body instead of
    the default `application/x-www-form-urlencoded` -- matching what
    `POST /context`/`PATCH /context/{id}` actually accept. Regression test
    for the bug QA found on #35's first pass (see
    `test_full_page_loads_json_enc_extension_script`'s docstring for the
    full mechanism); a full real-browser form submission isn't practical
    in this httpx-based harness, so this attribute-presence assertion is
    the practical regression guard.
    """
    unique = uuid.uuid4()
    create_response = await seeded_client.post(
        "/context",
        json={"label": f"json-enc check {unique}", "body": f"json-enc check body {unique}"},
    )
    assert create_response.status_code == 201

    response = await seeded_client.get("/context", headers={"HX-Request": "true"})
    assert response.status_code == 200
    # At least the add-item form plus one edit form (the item just
    # created above, which guarantees at least one user item exists to
    # render an edit form for).
    assert response.text.count('hx-ext="json-enc"') >= 2


async def test_get_context_plain_request_gets_json(seeded_client: AsyncClient) -> None:
    """No `HX-Request` header and a non-`text/html` `Accept` (including
    `httpx`'s default `*/*`) gets JSON, not HTML.
    """
    response = await seeded_client.get("/context")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert isinstance(response.json(), list)


@pytest.mark.parametrize("method", ["POST", "PATCH", "DELETE"])
async def test_write_endpoints_hx_request_header_gets_fragment(
    seeded_client: AsyncClient, method: str
) -> None:
    """`HX-Request: true` on `POST`/`PATCH`/`DELETE` gets back `text/html`,
    the re-rendered `partials/context_user_items.html` fragment, not
    JSON. Each case creates its own fresh row first (via plain JSON) so
    `PATCH`/`DELETE` have a real `kind="user"` id to act on.
    """
    unique = uuid.uuid4()
    create_response = await seeded_client.post(
        "/context", json={"label": f"HX fragment {unique}", "body": f"HX fragment body {unique}"}
    )
    assert create_response.status_code == 201
    item_id = create_response.json()["id"]

    if method == "POST":
        url = "/context"
        body: dict[str, str] | None = {
            "label": f"HX fragment 2 {unique}",
            "body": f"HX fragment body 2 {unique}",
        }
    elif method == "PATCH":
        url = f"/context/{item_id}"
        body = {"label": f"HX fragment updated {unique}"}
    else:
        url = f"/context/{item_id}"
        body = None

    response = await seeded_client.request(method, url, json=body, headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="user-context-items"' in response.text


async def test_write_endpoints_plain_request_gets_json(seeded_client: AsyncClient) -> None:
    """Without `HX-Request`, `POST` returns `201` JSON, `PATCH` returns
    `200` JSON, and `DELETE` returns `204` with no body.
    """
    unique = uuid.uuid4()
    create_response = await seeded_client.post(
        "/context", json={"label": f"JSON write {unique}", "body": f"JSON write body {unique}"}
    )
    assert create_response.status_code == 201
    assert create_response.headers["content-type"].startswith("application/json")
    item_id = create_response.json()["id"]

    update_response = await seeded_client.patch(
        f"/context/{item_id}", json={"body": f"JSON write body updated {unique}"}
    )
    assert update_response.status_code == 200
    assert update_response.headers["content-type"].startswith("application/json")

    delete_response = await seeded_client.delete(f"/context/{item_id}")
    assert delete_response.status_code == 204
    assert delete_response.content == b""
