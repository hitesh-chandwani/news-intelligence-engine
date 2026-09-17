# Technical Design — Event-Driven Intelligence & Notification System

**Companion to:** `plan.md` (product requirements)
**Stack:** Python monolith + PostgreSQL/pgvector + Free-Tier LLM (Google Gemini 2.0 Flash / OpenAI-compatible) + Local FastEmbed
**Status:** Approved for 100% Free / Zero-Cost MVP Stack.

---

## 1. Scope

Build the MVP pipeline described in `plan.md`:

> **Discover → Identify Event → Understand → Contextualize → Notify**, for the **Silver** commodity, for a single user.

This document covers architecture, data model, the processing pipeline, the LLM
strategy, the API/UI surface, and the build order. Discovery utilizes free RSS &
Google News RSS feeds with local extraction (trafilatura), with search adapters
deferred (see §7).

---

## 2. Architecture

A single deployable Python service (FastAPI process) plus a PostgreSQL database.
The service runs three things in one process:

1. **Web/API** — HTTP endpoints + server-rendered UI.
2. **Scheduler** — APScheduler firing the pipeline on an interval.
3. **Pipeline** — the discover→notify stages, also invokable on demand via an
   endpoint.

```mermaid
flowchart TB
    subgraph service["nie service (one process)"]
        API["FastAPI + Jinja/HTMX\n(API + UI)"]
        SCHED["APScheduler\n(interval trigger)"]
        PIPE["Pipeline runner\n(staged, idempotent)"]
        SCHED --> PIPE
        API -->|POST /pipeline/run| PIPE
    end

    subgraph ext["External & Local Services"]
        DISC["Discovery providers\n(Google News RSS / RSS feeds / stub)"]
        EXTRACT["Content extraction\n(trafilatura - local CPU)"]
        LLM["LLM (OpenAI-compatible)\n(Google AI Studio: Gemini 2.0 Flash)"]
        EMBED["FastEmbed (local ONNX bge-small-en-v1.5)"]
        NOTIF["Notifications\n(Telegram Bot + Resend Email)"]
    end

    PG[("PostgreSQL 16\n+ pgvector")]

    PIPE --> DISC
    PIPE --> EXTRACT
    PIPE --> LLM
    PIPE --> EMBED
    PIPE --> NOTIF
    API --> PG
    PIPE --> PG
```

Rationale for a monolith: single user, single Watch, iteration speed matters more
than horizontal scale. The pipeline stages are written as independent functions
with explicit state in the database, so lifting them into Celery/Temporal later
is mechanical if the product grows.

---

## 3. Tech stack (locked)

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | One language for pipeline + web; best LLM/scraping ecosystem |
| Package/deps | `uv` | Fast, reproducible, lockfile |
| Lint/format/type | Ruff, mypy | Standard, fast |
| Test | pytest | Standard |
| Web framework | FastAPI + Uvicorn | Async, Pydantic-native, small |
| UI | Jinja2 + HTMX (+ pico.css) | Server-rendered, no build step, stays in the monolith |
| Data models | SQLAlchemy 2.0 (async) + Alembic | Mature ORM + migrations |
| Database | PostgreSQL 16 + `pgvector` | One datastore for relational + vector |
| Scheduler | APScheduler (`AsyncIOScheduler`) | In-process, no extra infra for MVP |
| Retries | `tenacity` | Per-stage transient-failure handling |
| LLM | Google AI Studio `gemini-2.5-flash` via OpenAI-compatible endpoint | **100% Free** (15 RPM / 1,500 requests/day free tier); swappable to Groq / DeepSeek / Ollama via `.env`. Scheduled to retire 2026-10-16 — accepted, MVP ships before then; if the build slips, repoint `LLM_MODEL` to the then-current free Flash model |
| Embeddings | `fastembed` (`BAAI/bge-small-en-v1.5`, 384d) | **100% Free**; runs locally on CPU with ONNX Runtime (~5ms/doc); eliminates paid external embedding API calls |
| Extraction | `trafilatura` | **100% Free**; state-of-the-art local article body and metadata extraction |
| Discovery | `feedparser` + RSS & Google News RSS | **100% Free**; no paid search API required for MVP |
| Notifications | Telegram Bot API + Resend (email) | **100% Free**; Telegram provides instant push alerts on mobile/desktop; Resend free tier (3,000/mo) for email |
| Config | Pydantic Settings + `.env` | Typed config |
| Local/dev/deploy | Docker Compose (`db` + `app`) | One command up |

---

## 4. Data model

All tables carry `watch_id` so the schema is multi-Watch-ready (`plan.md` §16),
even though the MVP seeds exactly one Watch. Timestamps are `timestamptz`.

### `watch` — FR-001, FR-002
| Column | Type | Notes |
|---|---|---|
| id | uuid pk | |
| slug | text unique | `"silver"` |
| name | text | `"Silver Commodity"` |
| status | text | `enabled` \| `disabled` |
| created_at, updated_at | timestamptz | |

### `context_item` — FR-003, FR-004, FR-005
| Column | Type | Notes |
|---|---|---|
| id | uuid pk | |
| watch_id | uuid fk | |
| kind | text | `system` (seeded background) \| `user` (user-provided) |
| label | text | e.g. `"Supply factors"`, or user's own title |
| body | text | free text |
| created_at, updated_at | timestamptz | |

### `source` — FR-007
| Column | Type | Notes |
|---|---|---|
| id | uuid pk | |
| watch_id | uuid fk | |
| url | text | unique per watch; dedup key for discovery |
| title | text | |
| source_name | text | e.g. `"Reuters"` |
| published_at | timestamptz null | |
| discovered_at | timestamptz | |
| content | text null | full extracted text |
| extracted_at | timestamptz null | |
| extract_attempts | integer, default `0` | incremented per extraction attempt; caps retry of `extract_failed` rows at `MAX_EXTRACT_ATTEMPTS` |
| entities | jsonb | `["Fresnillo", "Mexico", "solar"]` |
| embedding | `vector(384)` null | over title + content summary (FastEmbed bge-small-en-v1.5) |
| status | text | `discovered` \| `extracted` \| `extract_failed` \| `triaged_out` \| `processed` |
| triage_note | text null | why triage dropped it, if it did |

### `event` — FR-010, FR-016, FR-017, FR-019
| Column | Type | Notes |
|---|---|---|
| id | uuid pk | |
| watch_id | uuid fk | |
| title | text | |
| fact_summary | text | **observed information only** |
| interpretation | text | **system interpretation, clearly separated** |
| event_date | timestamptz null | when it happened |
| discovered_at | timestamptz | |
| relevance | text | `irrelevant` \| `low` \| `medium` \| `high` (FR-012) |
| importance | text | `low` \| `medium` \| `high` \| `critical` (FR-014) |
| impact_direction | text | `bullish` \| `bearish` \| `neutral` \| `unclear` (FR-015) |
| impact_reason | text | |
| impact_confidence | text | `low` \| `medium` \| `high` |
| entities | jsonb | |
| embedding | `vector(384)` | for dedup / relatedness (FastEmbed bge-small-en-v1.5) |
| score_attempts | integer, default `0` | incremented per `score_stage` call attempt; caps retry of `relevance IS NULL` rows at `MAX_SCORE_ATTEMPTS` |
| related_at | timestamptz null | set once after a successful `relate_stage` pass for this event, regardless of outcome; never re-checked once set |
| last_material_update_at | timestamptz | bumped only on material change (FR-019) |
| created_at, updated_at | timestamptz | |

### `event_source` — FR-009 (many sources → one event)
`event_id` fk, `source_id` fk, `linked_at`. Unique `(event_id, source_id)`.

### `event_relation` — FR-018
`from_event_id` fk, `to_event_id` fk, `relation` text (`precedes` \| `similar` \| `escalation-of` \| `context-for`), `rationale` text.

### `category` + `event_category` — FR-011
`category`: `id`, `slug`, `name`. Seeded with the 13 categories from `plan.md`
§6; new rows can be added without code changes. `event_category`: `event_id` fk,
`category_id` fk (many-to-many).

### `notification_preference` — FR-023, FR-024
| Column | Type | Notes |
|---|---|---|
| watch_id | uuid fk unique | one row per watch |
| min_importance | text | `medium` default |
| categories | text[] | empty = all categories |
| channels | text[] | `["telegram", "email", "inapp"]` |

### `notification` — FR-020, FR-021, FR-022
| Column | Type | Notes |
|---|---|---|
| id | uuid pk | |
| watch_id, event_id | uuid fk | |
| reason | text | `new-event` \| `material-update` |
| payload | jsonb | rendered fields: title, what happened, why it matters, category, importance, impact, confidence, historical context, source list |
| channels_sent | text[] | |
| created_at | timestamptz | |
| read_at | timestamptz null | in-app read state |

### `feedback` — FR-025, FR-026
| Column | Type | Notes |
|---|---|---|
| id | uuid pk | |
| watch_id, event_id | uuid fk | |
| verdict | text | `useful` \| `not_useful` \| `too_many_similar` \| `more_like_this` \| `less_of_this` |
| note | text null | |
| created_at | timestamptz | |

### `pipeline_run` — observability
| Column | Type | Notes |
|---|---|---|
| id | uuid pk | |
| trigger | text | `schedule` \| `manual` |
| started_at, finished_at | timestamptz | |
| status | text | `running` \| `ok` \| `partial` \| `failed` |
| stats | jsonb | per-stage counts (discovered, extracted, new_events, updated_events, notified, errors) |
| error | text null | |

---

## 5. Pipeline

One run = these stages in order. Each stage reads/writes explicit row state and
is safe to re-run; a crashed run is recovered by simply running again. `tenacity`
wraps network calls. Every stage writes counters into `pipeline_run.stats`.

| # | Stage | Model | FRs | What it does |
|---|---|---|---|---|
| 1 | **discover** | — | FR-006, FR-007 | Each enabled `DiscoveryProvider` yields candidate items `{url, title, source_name, published_at, snippet, entities?}`. Drop items whose `url` already exists. Insert new `source` rows, `status=discovered`. |
| 2 | **extract** | — | FR-007 | For `discovered` sources without content, fetch full text using `TrafilaturaExtractor`. Set `content`, `extracted_at`, `status=extracted`. Failure → `status=extract_failed` (retried next run, capped). |
| 3 | **triage** | Gemini 2.0 Flash | FR-012 (cheap pre-filter) | Quick "could this plausibly be a meaningful Silver event?" Drops obvious noise → `status=triaged_out` + `triage_note`. Keeps deep reasoning calls for real candidates. |
| 4 | **embed** | FastEmbed | FR-008 | Embed `title + content` using local `bge-small-en-v1.5` on CPU (384-dim). Store `source.embedding`. |
| 5 | **match** | — | FR-008, FR-009 | pgvector cosine search over `event.embedding` for events within `DEDUP_WINDOW_DAYS`. Produce a candidate set (may be empty). |
| 6 | **adjudicate** | Gemini 2.0 Flash (structured) | FR-008, FR-009, FR-019 | Given the source + candidate events, decide: `new` \| `existing(event_id)` \| `noise`, plus `materiality: none \| minor \| material`. |
| 7 | **synthesize** | Gemini 2.0 Flash (structured) | FR-010, FR-011, FR-016 | `new` → build the full event record (title, **fact_summary vs interpretation**, event_date, entities, categories). `existing + material` → update the record and bump `last_material_update_at`. Always link `event_source`. |
| 8 | **score** | Gemini 2.0 Flash (structured) | FR-012, FR-013, FR-014, FR-015 | Assign `relevance`, `importance`, `impact_{direction,reason,confidence}`. Prompt is fed the retrieved context bundle (see §6). Failure → row stays `relevance IS NULL` (retried next run, capped). |
| 9 | **relate** | Gemini 2.0 Flash (structured) | FR-018 | Link to related historical events → `event_relation` rows. |
| 10 | **notify** | — | FR-019, FR-020–FR-024 | Emit a `notification` iff: run produced a `new` event **or** a `material` update; **and** `relevance != irrelevant`; **and** `importance >= pref.min_importance`; **and** (`pref.categories` empty or event shares one). Render `payload`, send on `pref.channels` (Telegram / email), record `channels_sent`. |

Stages 6–9 can be merged into fewer LLM calls during tuning; they are listed
separately for clarity. `DEDUP_WINDOW_DAYS`, poll interval, and the importance
threshold are config (§17).

### Idempotency / recovery
- Discovery dedups on `source.url`.
- Each stage filters on row `status` / null columns, so re-running skips
  completed work.
- `event_source` and `event_relation` have unique constraints.
- A partial run leaves rows in intermediate states; the next run finishes them.

---

## 6. LLM strategy

### OpenAI-Compatible Provider Architecture
To ensure zero cost and prevent vendor lock-in, the LLM client uses the standard **OpenAI-compatible protocol** (`openai` Python SDK or `httpx`):
- **Default / Primary Model:** Google AI Studio `gemini-2.5-flash`
  - Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
  - **Free Tier:** 15 Requests Per Minute (RPM), 1,500 Requests Per Day (RPD), 1M token context.
  - Zero cost, high speed, top-tier benchmark reasoning.
  - **Retirement risk:** Google has rotated the free-tier flagship Flash model every 6–8 weeks through 2026; `gemini-2.5-flash` itself is scheduled to retire 2026-10-16. Accepted for this MVP given the build timeline. Because the client speaks the OpenAI-compatible protocol, recovering from an unannounced retirement is a one-line `.env` change (`LLM_MODEL`), not a code change — confirm the current free Flash model ID in Google AI Studio before M3.
- **Alternative Free / Low-Cost Providers (Swappable via `.env`):**
  - **Groq:** `llama-3.3-70b-versatile` (generous free tier, ultra-fast).
  - **DeepSeek:** `deepseek-chat` / V3 (~$0.14/1M tokens).
  - **Local Ollama:** `http://localhost:11434/v1` (`qwen2.5:14b` or `llama3.2:3b`, 100% offline).

### Judgment quality on a small free model
Unlike a two-tier design, every reasoning stage — including dedup adjudication
and the relevance/importance/impact scoring that fact/interpretation
separation depends on (FR-016) — now runs on the same small free model, not
just triage. This is the part of the pipeline that determines whether
notifications are worth reading. The eval set in §16 is the guardrail: if it
shows the free model is unreliable on adjudication or impact reasoning,
escalate only stages 6–9 to a stronger model (still free/cheap-tier — e.g. a
Groq 70B model) before shipping, rather than discovering it from bad
notifications in production.

### Rate Limit & Throttle Management
Because Google AI Studio Free Tier has a 15 RPM limit:
- The pipeline runner includes an async token bucket rate limiter — spaced at
  ~5 seconds between LLM calls (not the exact 4s = 15 RPM boundary, so normal
  jitter doesn't trip 429s), with backoff-and-retry on `429` regardless.
- Triage pre-filters noise so only viable candidate articles consume LLM calls.
- Typical hourly run of 15–20 articles uses ~20–40 requests, well inside the 1,500/day limit.

### Structured outputs
Use Pydantic models with JSON Schema validation (`response_format={"type": "json_object"}` or schema definition) for each of:
`AdjudicationResult`, `EventRecord`, `ScoreResult`, `RelationSet`. No loose prose
parsing. `fact_summary` and `interpretation` are separate required fields in
`EventRecord` — the schema strictly enforces FR-016.

JSON-schema adherence through the OpenAI-compat shim is less proven than a
provider's native structured-output path (true of every free-tier provider in
the swap list, not just Google). Wrap every parse in validate-then-retry: on a
schema-invalid response, re-ask once with the validation error appended before
failing the stage — cheap insurance against an occasional malformed response.

### Context bundle (FR-013)
The retrieval that feeds stage 8:
- all `context_item` rows for the watch (system + user),
- `event_relation`-linked and vector-nearest historical events,
- the source's and candidate event's `entities`,
- a feedback summary: counts of each `verdict` in the last N days, bucketed by
  category (e.g. "user marked 4 `Market` events `too_many_similar`").

### Prompt inventory (files under `src/nie/llm/prompts/`)
`system.md`, `rubric.md`, `triage.md`, `adjudicate.md`, `synthesize.md`,
`score.md`, `relate.md`, `notification_render.md`.

---

## 7. Discovery abstraction (source deferred)

```python
class DiscoveryProvider(Protocol):
    name: str
    def discover(self, watch: Watch, since: datetime) -> Iterable[CandidateItem]: ...
```

`CandidateItem = {url, title, source_name, published_at, snippet, content?, entities?}`

Providers shipped in build order:
1. **`StubProvider`** — reads `tests/fixtures/sources/*.json`. Lets the whole
   pipeline run offline during development and in CI.
2. **`RssProvider`** — curated industry feeds (Kitco, Mining.com, Reuters) plus
   **Google News RSS query** (`https://news.google.com/rss/search?q=silver+commodity+mining`).
   100% Free, deterministic, real-time web discovery without paid search APIs.
   Note: Google News RSS entries are `news.google.com/rss/articles/...`
   redirect links, not the publisher URL — resolve the redirect before handing
   the URL to `TrafilaturaExtractor`. A resolution failure lands in
   `extract_failed` and is retried next run like any other extraction failure.
3. **`WebSearchProvider`** — (Optional / Deferred plugin) adapter for external search
   APIs (e.g. DuckDuckGo / Tavily / Brave) if broader search coverage is needed post-MVP.

`DISCOVERY_PROVIDERS` (config, CSV) selects which are active. Content extraction
uses `TrafilaturaExtractor` (local CPU, zero cost). A provider that already returns
`content` skips extraction.

---

## 8. Dedup & historical linking

- **Candidate recall:** pgvector cosine `ORDER BY embedding <=> :q LIMIT k`,
  filtered to events with `event_date` (or `discovered_at`) inside
  `DEDUP_WINDOW_DAYS`. Keeps the LLM adjudication input small.
- **Decision:** stage 6 LLM call — `new` / `existing` / `noise` + materiality.
- **Same event, new reporting (FR-019):** `existing` + `materiality != material`
  → link the `source` to the event, do **not** bump
  `last_material_update_at`, do **not** notify.
- **Relatedness (FR-018):** stage 9 proposes `event_relation` rows against
  vector-nearest + entity-overlapping historical events.

---

## 9. Scoring & fact/interpretation split

- `relevance` — `irrelevant` short-circuits: no notification, event still
  stored for history.
- `importance` — potential significance to the Silver market or the user's
  stated interests (user context items are in the prompt).
- `impact` — `{direction, reason, confidence}`; `unclear` is a valid direction.
- `fact_summary` vs `interpretation` — enforced as separate schema fields and
  rendered as separate labelled blocks in the UI and the notification.

---

## 10. Notification

- **Trigger gate:** see stage 10 in §5.
- **Content (FR-021):** `payload` holds event title, "what happened"
  (`fact_summary`), "why it matters" (`interpretation` + importance rationale),
  category, importance, impact + confidence, related historical context (from
  `event_relation`), and the full source list with URLs (FR-022).
- **Channels:**
  - `telegram` via Telegram Bot API (100% free, instant push alert to mobile/desktop).
  - `email` via Resend (Jinja-rendered HTML, free tier 3,000/mo).
  - `inapp` = `notification` rows shown in the UI inbox.
  `pref.channels` selects active channels.
- **Preferences (FR-023, FR-024):** `min_importance` and `categories` on
  `notification_preference`, editable in the UI.

---

## 11. Feedback loop

- UI shows the 5 verdicts from `plan.md` §13 on every event/notification.
- `POST /events/{id}/feedback` writes a `feedback` row.
- The pipeline's context bundle (§6) includes a rolling feedback summary, so
  future scoring and the notify gate see it (FR-026). MVP keeps this in-prompt;
  no fine-tuning.

---

## 12. HTTP API

| Method + path | Purpose | FRs |
|---|---|---|
| `GET /` | Dashboard: watch status, recent events, unread notifications | — |
| `POST /watch/enable` / `POST /watch/disable` | Toggle monitoring | FR-002 |
| `GET /watch/status` | Current status + last run summary | FR-002 |
| `GET /context` / `POST /context` / `PATCH /context/{id}` / `DELETE /context/{id}` | User context CRUD | FR-004, FR-005 |
| `GET /events` | List/filter by category, importance, relevance, date | FR-027 |
| `GET /events/{id}` | Event detail: fact/interpretation, impact, sources, related events | FR-027 |
| `GET /notifications` | In-app inbox; `POST /notifications/{id}/read` | FR-020 |
| `POST /events/{id}/feedback` | Submit feedback | FR-025 |
| `DELETE /events/{id}/feedback/{feedback_id}` | Withdraw feedback | FR-025 |
| `GET /preferences` / `PATCH /preferences` | Importance threshold + categories | FR-023, FR-024 |
| `POST /pipeline/run` | Trigger a run now | — |
| `GET /pipeline/runs` | Run history + stats | — |
| `POST /pipeline/runs/{run_id}/cancel` | Cancel a running run + release the pipeline lock | — |

## 13. UI pages (Jinja + HTMX)

- **Dashboard** — status toggle, last run, recent events, unread notifications.
- **Timeline** — filterable event list; each row links to detail.
- **Event detail** — labelled Fact / Interpretation blocks, impact + confidence,
  categories, source links, related events, feedback buttons.
- **Context editor** — list/add/edit/remove user context items; system context
  shown read-only.
- **Preferences** — importance threshold, category multi-select, channels.
- **Inbox** — notifications, mark-read.

---

## 14. Configuration

`.env` (via Pydantic Settings):

| Var | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgrespassword@localhost:5432/nie_db` | Postgres connection string |
| `LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | OpenAI-compatible endpoint (Google AI Studio) |
| `LLM_API_KEY` | — | Google AI Studio free API key (or Groq/DeepSeek) |
| `LLM_MODEL` | `gemini-2.5-flash` | Primary model for triage and adjudication (retires 2026-10-16 — verify current free Flash model before M3) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model (runs locally on CPU, 384d) |
| `TELEGRAM_BOT_TOKEN` | — | Telegram Bot API token (free instant push alerts) |
| `TELEGRAM_CHAT_ID` | — | Telegram target chat / user ID |
| `RESEND_API_KEY` | — | Optional: Resend API key for email delivery |
| `NOTIFY_EMAIL_TO` | — | Optional: Email recipient |
| `DISCOVERY_PROVIDERS` | `stub,rss` | CSV: `stub`, `rss` |
| `RSS_FEEDS` | — | CSV of RSS feeds (Google News Silver query + industry feeds) |
| `POLL_INTERVAL_MINUTES` | `60` | Scheduler cadence |
| `NOTIFY_MIN_IMPORTANCE` | `medium` | Seed for `notification_preference` |
| `DEDUP_WINDOW_DAYS` | `14` | Candidate recall window |
| `MAX_EXTRACT_ATTEMPTS` | `3` | Cap on `extract` stage retries per `source` row before it stops being retried |
| `MAX_SCORE_ATTEMPTS` | `3` | Cap on `score` stage retries per `event` row before it stops being retried |

---

## 15. Project layout

```
news-intelligence-engine/
  pyproject.toml            uv.lock
  docker-compose.yml        Dockerfile        .env.example
  alembic.ini               alembic/
  src/nie/
    config.py               db.py             models.py          schemas.py
    web/
      app.py  routers/  templates/  static/
    pipeline/
      runner.py  discover.py  extract.py  triage.py  embed.py
      match.py   adjudicate.py  synthesize.py  score.py  relate.py  notify.py
    llm/
      client.py  prompts/
    sources/
      base.py  stub.py  rss.py  extract_trafilatura.py
    embeddings/   fastembed.py
    notifications/ telegram.py  email.py  inbox.py
    seed/         silver_context.md  categories.py  run.py
  tests/
    fixtures/sources/*.json
    test_pipeline_*.py  eval/  (labelled cases for §16)
```

---

## 16. Evals

The plan's success criteria 2–5 and 9–10 are classification-quality targets, so
build a small labelled set early:

- 30–50 fixture articles tagged with: new-event vs same-event (vs which event),
  expected relevance, expected importance.
- A pytest target runs stages 3–8 against fixtures and reports agreement.
- Used to gate any prompt change and any model-tier downgrade.

---

## 17. Deployment

`docker-compose.yml`:
- `db` — `pgvector/pgvector:pg16`, volume, healthcheck.
- `app` — the `nie` image; `depends_on: db`; runs Alembic migrations on start,
  then Uvicorn (which also starts the APScheduler job).

`uv run` / `make` tasks: `migrate`, `seed`, `dev`, `run-once` (one pipeline
run), `test`, `eval`.

Seed (`src/nie/seed/run.py`): create the Silver `watch`, load `silver_context.md`
into `context_item` rows (`kind=system`), insert the 13 categories, insert the
default `notification_preference`.

---

## 18. Resolved Architecture Decisions (Free MVP)

| # | Decision | Selected Choice | Rationale |
|---|---|---|---|
| 1 | UI approach | FastAPI + Jinja + HTMX (in-monolith) | Zero build step, minimal footprint, fast prototyping |
| 2 | LLM Provider & Model | Google AI Studio (`gemini-2.5-flash`) via OpenAI-compatible endpoint | **100% Free** (1,500 requests/day, 15 RPM). Retires 2026-10-16, accepted given MVP timeline; `.env`-swappable if it slips |
| 3 | Embeddings provider | `fastembed` (`bge-small-en-v1.5`, 384d) | **100% Free**, runs locally on CPU with ONNX Runtime, no external API latency |
| 4 | MVP notification channels | Telegram Bot API + Resend (email) + in-app | Telegram is 100% free with instant phone/desktop push; Resend free tier for email |
| 5 | Auth | None for MVP | Single-user, bound to localhost / trusted network |
| 6 | Discovery sources | Curated industry RSS + Google News RSS query | Real-time web discovery with 0 API costs |

---

## 19. Build order

| Milestone | Deliverable |
|---|---|
| M0 | Repo skeleton, `uv` deps, Docker Compose (`pgvector`), config, `GET /health` |
| M1 | `models.py`, first Alembic migration (384d vectors), seed script (watch + context + categories + prefs) |
| M2 | Pipeline spine: `pipeline_run` tracking, `runner.py`, `POST /pipeline/run`, `StubProvider` |
| M3 | LLM stages 6–9 using OpenAI-compatible client (`gemini-2.5-flash`, verify still current) + structured outputs + validate/retry wrapper; eval fixtures |
| M4 | pgvector wiring, `FastEmbed` embeddings + vector match stage, triage stage with rate-limiting |
| M5 | Notifications: Telegram Bot push + Resend email + in-app inbox |
| M6 | UI: dashboard, timeline, event detail, context editor, preferences, inbox |
| M7 | Feedback capture + rolling feedback summary into the context bundle |
| M8 | `RssProvider` (Google News RSS query + Kitco/Mining.com) + `TrafilaturaExtractor` |
| M9 | APScheduler interval automation on `POLL_INTERVAL_MINUTES` |

---

## 20. Out of scope (from `plan.md` §15)

Price prediction, buy/sell recommendations, automated trading, portfolio
management, technical strategies, a complex dashboard, multiple Watch types,
non-Silver monitoring, autonomous long-form research.

**Extensibility kept in:** every table is keyed by `watch_id`; the pipeline
takes a `Watch`; discovery/context/scoring are watch-scoped. Adding a second
Watch later is data + config, not a redesign.
