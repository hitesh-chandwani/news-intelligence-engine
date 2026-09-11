# Backlog

Companion to `plan.md` (requirements) and `design.md` (architecture). Each task
below is scoped to one session and self-contained: it names the concrete
tables/files/endpoints involved and points at the relevant `design.md`
section, so nobody needs to have read another task in this file to pick one
up — just `plan.md`/`design.md` and the current state of the repo. Tasks are
ordered so that doing them in sequence builds the system incrementally, but
each one's description stands on its own.

---

**Foundation**

## 1. Project skeleton with a passing test
Goal: Get an empty, installable Python project with working tooling and one passing test.
Description: Initialize the `uv`-managed project (`pyproject.toml`, `src/nie/` package, `tests/`), add Ruff/mypy/pytest config, and commit a single trivial test that passes (e.g. `test_placeholder`). No application logic yet — this only proves the toolchain works end to end.

## 2. Docker Compose for Postgres + pgvector
Goal: Bring up a local Postgres instance with the pgvector extension in one command.
Description: Add `docker-compose.yml` with a `db` service on the `pgvector/pgvector:pg16` image, a named volume, and a healthcheck. Document `docker compose up db` in the README and confirm `CREATE EXTENSION vector;` succeeds against it.

## 3. App configuration module
Goal: Centralize all environment-driven configuration in one typed settings object.
Description: Add `src/nie/config.py` using Pydantic Settings, covering every variable listed in `design.md` §14 (DB URL, LLM base URL/key/model, embedding model, Telegram/Resend keys, discovery providers, polling interval, importance threshold, dedup window). Commit a `.env.example` with all keys present but unset, and a test that loading settings from a fixture `.env` produces the expected values.

## 4. Database engine + Alembic wiring
Goal: Get a working async DB connection and an empty Alembic migration chain.
Description: Add `src/nie/db.py` (async SQLAlchemy engine/session factory reading `DATABASE_URL` from config) and initialize Alembic (`alembic.ini`, `alembic/env.py` wired to the async engine). Commit one no-op migration and a test that runs migrations against the Docker Compose database from task 2 and confirms the connection works.

---

**Data model** (`design.md` §4)

## 5. `watch` table
Goal: Persist a Watch as the top-level container the rest of the schema hangs off.
Description: Add the `watch` SQLAlchemy model and an Alembic migration (id, slug, name, status, timestamps). Add a test that creates and reads back a watch row with `status='enabled'`.

## 6. `context_item` table
Goal: Persist Silver background context and user-provided context against a Watch.
Description: Add the `context_item` model and migration (id, watch_id fk, kind `system`/`user`, label, body, timestamps). Add a test inserting one `system` and one `user` row for a watch and reading them back.

## 7. Category taxonomy tables
Goal: Persist the event category taxonomy as data, not code.
Description: Add `category` and `event_category` models and migration, plus a seed function inserting the 13 categories from `plan.md` §6. Add a test confirming the seed function is idempotent (running it twice doesn't duplicate rows).

## 8. `source` table
Goal: Persist a discovered piece of web content and its processing status.
Description: Add the `source` model and migration (url unique per watch, title, source_name, published/discovered timestamps, content, entities jsonb, 384-dim embedding column, status enum, triage_note). Add a test that inserting two sources with the same URL for the same watch is rejected by the unique constraint.

## 9. `event` table
Goal: Persist an identified event with its fact/interpretation split and scoring fields.
Description: Add the `event` model and migration, including the separate `fact_summary`/`interpretation` columns (FR-016), relevance/importance/impact fields, entities, a 384-dim embedding column, and `last_material_update_at`. Add a test that round-trips a fully populated event row.

## 10. `event_source` and `event_relation` tables
Goal: Support many-to-many linking between events, their sources, and other events.
Description: Add both join tables and their migration, with a unique constraint on `event_source(event_id, source_id)`. Add a test linking one event to two sources and to one related event, then querying both relationships back.

## 11. Notification preference and notification tables
Goal: Persist per-watch notification preferences and the notifications that get sent.
Description: Add `notification_preference` (one row per watch: min_importance, categories, channels) and `notification` (payload jsonb, channels_sent, read_at) models and migration. Add a test creating a default preference row and one notification referencing an event.

## 12. `feedback` table
Goal: Persist user feedback on an event or notification.
Description: Add the `feedback` model and migration (watch_id, event_id, verdict enum of the 5 values in `plan.md` §13, note, created_at). Add a test inserting one row per verdict value.

## 13. `pipeline_run` table
Goal: Give every pipeline execution a durable, queryable record.
Description: Add the `pipeline_run` model and migration (trigger, started_at/finished_at, status, stats jsonb, error). Add a test that creates a run, updates it to `status='ok'` with a stats payload, and reads it back.

## 14. Seed script
Goal: Get from an empty database to a usable Silver watch in one command.
Description: Add `src/nie/seed/run.py` that creates the Silver watch, loads a `silver_context.md` content file into system `context_item` rows, seeds the 13 categories (reusing task 7's seed function), and inserts a default `notification_preference`. Add a test that running the seed script twice is idempotent.

---

**Discovery & extraction** (`design.md` §7 — offline-testable, no live network in tests)

## 15. Discovery provider interface + stub provider
Goal: Define the discovery contract and a fixture-based implementation nothing else needs the network for.
Description: Add the `DiscoveryProvider` protocol and `CandidateItem` type from `design.md` §7, plus a `StubProvider` that reads `tests/fixtures/sources/*.json`. Add a test asserting the stub yields the expected candidate items from a small fixture set.

## 16. Content extraction interface + trafilatura extractor
Goal: Turn a URL into clean article text.
Description: Add the `Extractor` protocol and a `TrafilaturaExtractor` implementation that fetches a URL and returns extracted title/text, raising a typed error on failure. Add a test against a saved local HTML fixture (no live network call in the test).

## 17. RSS discovery provider
Goal: Discover real Silver-related articles from RSS feeds without a paid API.
Description: Add `RssProvider` using `feedparser` against a configurable feed list (curated industry feeds plus a Google News RSS query, per `design.md` §7), resolving `news.google.com/rss/articles/...` redirect links to the publisher URL before returning them. Add a test against a saved local feed fixture.

---

**Pipeline spine** (`design.md` §5)

## 18. Pipeline runner scaffolding
Goal: Give the pipeline stages a shared place to run, track progress, and be triggered.
Description: Add `src/nie/pipeline/runner.py`: a stage-execution loop that opens a `pipeline_run` row, calls each registered stage function in order, records per-stage counts into `stats`, and marks the run `ok`/`partial`/`failed`. Wire stages as a list of no-op placeholders for now. Add a test that a run with zero real stages completes and is recorded.

## 19. Discover stage
Goal: Turn active discovery providers into new `source` rows.
Description: Implement the `discover` stage: call each provider named in `DISCOVERY_PROVIDERS`, drop items whose URL already exists for the watch, and insert the rest as `source` rows with `status='discovered'`. Add a test using the `StubProvider` from task 15 confirming re-running discovery doesn't duplicate sources.

## 20. Extract stage
Goal: Fill in full article content for newly discovered sources.
Description: Implement the `extract` stage: for `discovered` sources without content, run the configured extractor, setting `content`/`extracted_at`/`status='extracted'` on success or `status='extract_failed'` on failure. Add a test covering both the success and failure paths against fixture sources.

---

**Embeddings & dedup** (`design.md` §5, §8)

## 21. Local embeddings module + embed stage
Goal: Compute a vector representation for a source or event's text.
Description: Add `src/nie/embeddings/fastembed.py` wrapping `fastembed`'s `bge-small-en-v1.5` model (documenting the one-time model download), and an `embed` stage that fills `source.embedding` for `extracted` sources. Add a test that embedding the same text twice produces the same 384-dimension vector.

## 22. Vector match stage
Goal: Find candidate existing events a new source might belong to.
Description: Implement the `match` stage: given a source's embedding, run a pgvector cosine-distance query over `event.embedding` restricted to events within `DEDUP_WINDOW_DAYS`, returning a ranked candidate list. Add a test with three seeded events (two similar, one outside the window) confirming the right candidate set comes back.

---

**LLM stages** (`design.md` §5, §6)

## 23. LLM client wrapper
Goal: One shared, resilient way for every pipeline stage to call the LLM.
Description: Add `src/nie/llm/client.py`: an OpenAI-compatible client configured from `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`, a rate limiter spaced comfortably under the free-tier RPM limit, and a `call_structured()` helper that validates the response against a Pydantic schema and retries once with the validation error appended on failure. Add a test against a stubbed HTTP response covering the retry path.

## 24. Triage stage
Goal: Cheaply drop sources that are obviously not a meaningful Silver event before spending a full reasoning call on them.
Description: Add the triage prompt and stage: call the LLM client with a source's title/content and ask for a plausibility verdict, setting `status='triaged_out'` + `triage_note` on a negative verdict. Add a test with a stubbed LLM response covering both verdicts.

## 25. Adjudication stage
Goal: Decide whether a source describes a new event, an existing one, or is noise.
Description: Add the adjudication prompt and stage: given a source and its candidate events (from task 22), get a structured `new` / `existing(event_id)` / `noise` decision plus a `materiality` level. Add a test with a stubbed LLM response for each of the three decisions.

## 26. Synthesis stage
Goal: Create or update the event record for an adjudicated source.
Description: Add the synthesis prompt and stage: for a `new` decision, build the full event record (title, `fact_summary`, `interpretation`, event_date, entities, categories); for an `existing` + material decision, update the record and bump `last_material_update_at`; always link the source via `event_source`. Add a test covering both the create and update paths with stubbed LLM output.

## 27. Context bundle builder
Goal: Assemble the retrieval context every judgment-stage prompt needs.
Description: Add a function that, given a watch and a candidate event, returns the system + user `context_item` rows, vector-nearest and relation-linked historical events, shared entities, and a rolling feedback summary bucketed by category and verdict, per `design.md` §6. Add a test with seeded fixture data asserting the bundle contains the expected pieces.

## 28. Scoring stage
Goal: Assign relevance, importance, and impact to an event.
Description: Add the scoring prompt and stage: call the LLM with the event and the context bundle from task 27 to get structured `relevance`, `importance`, and `impact_{direction,reason,confidence}` values written onto the event. Add a test with a stubbed LLM response covering at least one value from each enum.

## 29. Historical relation stage
Goal: Link a new or updated event to related past events.
Description: Add the relation prompt and stage: given an event and its nearest historical candidates, get a structured set of `event_relation` rows (relation type + rationale) and persist them. Add a test with a stubbed LLM response creating two relations.

---

**Notification** (`design.md` §10)

## 30. Notification gate + payload builder
Goal: Decide whether an event deserves a notification and build its content.
Description: Add a function implementing the notify gate from `design.md` §5 stage 10 (new-or-material, relevance not irrelevant, importance at or above the preference threshold, category match) and, when it passes, build the `notification.payload` fields listed in `design.md` §10. Add a test covering one event that passes the gate and one that's correctly suppressed by each individual condition.

## 31. Email notification channel
Goal: Deliver a notification by email.
Description: Add `src/nie/notifications/email.py` using the Resend SDK and a Jinja template rendering the notification payload into an HTML email. Add a test that renders the template against a fixture payload and asserts the key fields (title, fact, interpretation, sources) appear in the output.

## 32. Telegram notification channel
Goal: Deliver a notification as an instant push message.
Description: Add `src/nie/notifications/telegram.py` using the Telegram Bot API to send a formatted message to `TELEGRAM_CHAT_ID`. Add a test that renders the message text from a fixture payload against a stubbed HTTP call (no live send).

## 33. In-app notification inbox data layer
Goal: Make sent notifications queryable and mark them read.
Description: Add functions for listing a watch's notifications (newest first, with read/unread state) and marking one read, backing the future inbox UI. Add a test covering list ordering and the read-state transition.

---

**API & UI** (`design.md` §12–13)

## 34. Watch status API + UI
Goal: Let the user see and toggle whether the Silver watch is monitoring.
Description: Add `GET /watch/status`, `POST /watch/enable`, `POST /watch/disable`, and a minimal Jinja page showing the current status with a toggle button. Add a test exercising enable → status → disable through the API.

## 35. Context editor API + UI
Goal: Let the user view system context and manage their own context items.
Description: Add `GET/POST /context`, `PATCH/DELETE /context/{id}`, and a page listing system context read-only with user context items editable inline (FR-004, FR-005). Add a test covering create, update, and delete of a user context item.

## 36. Preferences API + UI
Goal: Let the user set the notification threshold, category filter, and channels.
Description: Add `GET/PATCH /preferences` and a page editing `min_importance`, a category multi-select, and channel checkboxes (FR-023, FR-024). Add a test that a PATCH updates and persists all three fields.

## 37. Event timeline + detail API + UI
Goal: Let the user browse and inspect the events the system has found.
Description: Add `GET /events` (filterable by category/importance/relevance/date) and `GET /events/{id}`, plus timeline and detail pages showing the labelled Fact/Interpretation blocks, impact, sources, and related events (FR-027). Add a test that filtering by importance returns only matching events.

## 38. Notification inbox UI
Goal: Give the user a place to see and clear notifications in the app.
Description: Add `GET /notifications` and `POST /notifications/{id}/read`, plus a page listing them with an unread indicator, built on task 33's data layer. Add a test that reading a notification through the route updates its `read_at`.

## 39. Feedback capture
Goal: Let the user tell the system whether an event or notification was useful.
Description: Add `POST /events/{id}/feedback` and feedback buttons for the 5 verdicts from `plan.md` §13 on the event detail and inbox pages. Add a test that submitting each verdict creates the corresponding `feedback` row.

---

**Operations**

## 40. Scheduler wiring
Goal: Run the pipeline automatically instead of only on manual trigger.
Description: Add APScheduler (`AsyncIOScheduler`) started alongside the FastAPI app, firing the pipeline runner every `POLL_INTERVAL_MINUTES`, and skipping the run entirely when the watch is disabled. Add a test that a disabled watch produces no `pipeline_run` and an enabled one does, using a short interval.

## 41. Eval harness
Goal: Measure whether the LLM stages classify events well enough to trust.
Description: Add 30–50 labelled fixture articles (new-event/same-event, expected relevance, expected importance) per `design.md` §16, and a pytest/CLI target that runs the triage/adjudicate/synthesize/score stages against them and reports agreement against the labels. Add a README note on the pass bar this gates future prompt or model changes against.
