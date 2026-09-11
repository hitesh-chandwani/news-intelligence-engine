# Testing guidelines

Companion to `process.md`. Applies to every task in `tasks.md`.

- **Framework:** `pytest`, run via `uv run pytest` (also exposed as the `test`
  task in task 41's `uv run`/`make` targets). Use `pytest-asyncio` for any
  test that touches the async DB engine or async pipeline stages.
- **No live network calls in tests.** Anything that would hit a real URL,
  RSS feed, LLM API, Telegram, or Resend must be exercised against a saved
  local fixture or a stubbed HTTP response instead (see tasks 15-17, 23-26,
  31-32). This keeps the suite fast, deterministic, and runnable with no
  credentials.
- **Fixtures live under `tests/fixtures/`**, mirroring the module they back
  (e.g. `tests/fixtures/sources/*.json` for discovery/extraction). Prefer a
  small, hand-authored fixture over a captured full response.
- **File naming:** one test module per source module, `test_<module>.py`.
  Pipeline-stage tests follow `test_pipeline_*.py` per `design.md` §15.
- **Database-backed tests** run against the Docker Compose Postgres instance
  from task 2 (`docker compose up db`), never mocked — matches this
  project's earlier decision that mocking the DB hides real migration/query
  bugs. Each test is responsible for the state it needs; don't depend on
  ordering between tests.
- **Idempotency matters.** Where a task's acceptance criteria calls for
  idempotency (seed functions, discovery re-runs), the test must call the
  function/stage twice and assert no duplication, not just that it runs once
  without error.
- **Structured LLM output tests** stub the HTTP layer and assert both the
  happy path and at least one failure/retry path (task 23's validation-retry
  behavior; tasks 24-26, 28-29's per-verdict/per-enum coverage).
- Keep tests trivial and focused: one behavior per test, no shared mutable
  fixtures across unrelated tests, no over-mocking internal collaborators
  that aren't the actual system boundary (network, filesystem, LLM API).
