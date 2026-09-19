# news-intelligence-engine

## Database

Bring up a local Postgres instance with the pgvector extension:

```sh
docker compose up db
```

This starts Postgres 16 with pgvector on `localhost:5432` (user `postgres`,
password `postgrespassword`, database `nie_db`), backed by a named volume
so data survives a restart. Wait for the container to report healthy, then
enable the extension once per database:

```sh
docker compose exec db psql -U postgres -d nie_db -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

## Embeddings

Source embeddings are generated locally with
[FastEmbed](https://github.com/qdrant/fastembed) running the
`BAAI/bge-small-en-v1.5` model on CPU (`EMBEDDING_MODEL`, see
`src/nie/config.py`) -- no external embeddings API or API key is
required.

The first call to `embed_text`/`embed_texts` in a process (so the first
`embed` pipeline stage run, and the first run of
`tests/test_embeddings_fastembed.py`) downloads the ~100MB
`BAAI/bge-small-en-v1.5` ONNX model from Hugging Face to fastembed's
local cache directory. This needs network access once and no
credentials (it's a public model); every run after that, on the same
machine, is fully offline as long as that cache directory persists.

## Eval harness

`tests/test_eval_harness.py` (issue #41, `_docs/design.md` §16) measures
judgment quality on the configured LLM: it runs the real pipeline stages
3-8 (`triage`, `embed`, `match`, `adjudicate`, `synthesize`, `score`)
against 36 hand-authored, hand-labelled fixture articles under
`tests/fixtures/eval/*.json`, and reports how often the pipeline's real
decisions agree with each fixture's expected `new_event`/`existing_event`
decision (and which other fixture's event it should match), `relevance`,
and `importance`.

**This is a real-LLM-call harness, run manually, distinct from
`uv run pytest`.** It is gated behind the `eval` pytest marker
(`pyproject.toml`'s `addopts = "-m 'not eval'"` deselects it from every
default `uv run pytest` invocation -- see `tests/test_eval_harness.py`'s
module docstring), and it makes real calls to whatever provider
`LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` point at. It exits with a clear,
actionable error instead of attempting a call if `LLM_API_KEY` isn't set.

Run it against a Compose Postgres instance (`docker compose up db -d`),
with a real `LLM_API_KEY` configured (`.env` or exported in the shell):

```sh
uv run pytest tests/test_eval_harness.py -m eval -s
```

`-s` is required to see the printed report (pytest captures stdout by
default). This is used to gate any future prompt change or model-tier
downgrade (`_docs/design.md` §6/§16) against a known-good baseline.

### Pass bar

First real run (2026-09-19), against `gemini-3.5-flash-lite` (Google AI
Studio free tier -- the originally-documented default,
`gemini-2.5-flash`, and every other Flash-tier alias tried before it,
`gemini-2.0-flash`/`gemini-2.5-flash-lite`, had already been retired for
new callers by this date; `gemini-3.5-flash-lite` was the smallest/
cheapest model still available), over all 36 fixtures under
`tests/fixtures/eval/`:

- **New/existing-event decision agreement: 80.6% (29/36)**
- **Existing-event → correct matched-event agreement: 100.0% (10/10)**
  -- every time the pipeline correctly called a fixture "existing", it
  matched it to the right prior event, no misattributions
- **Relevance exact-match agreement: 55.2% (16/29)**
- **Importance exact-match agreement: 48.3% (14/29)**

Read together with the per-fixture output (`-s`), the 7 decision misses
(29/36 correct) break into two different kinds:

- 3 of them (`29`, `34`, `36`) are fixtures deliberately written with
  `expected.relevance = "irrelevant"` -- the pipeline correctly kept all
  three out of the notification path (`triaged_out`/`noise`), which is
  the behavior that actually matters in production; they only count as
  "decision" misses because the fixture format's `decision` field is
  `new_event` even for an irrelevant article (there's no separate
  "irrelevant" decision value), not because the pipeline got anything
  wrong.
- The other 4 (`20`, `23`, `26`, `30`) are genuine misses: articles
  expected to become a real event (non-irrelevant) that the pipeline
  called `noise` instead -- these would have been silently dropped in
  production and are worth a closer look if this number doesn't improve
  on a future run.

Separately, the relevance/importance misses are mostly a *severity*
skew, not a *direction* skew: the model very rarely called something
relevant "irrelevant" or vice versa, but frequently over-rated a
fixture's `relevance`/`importance` by one notch (e.g. "medium" expected,
"high" returned) -- normal disagreement to expect from a small,
free-tier model on a subjective severity judgment, and the kind of thing
this harness exists to make visible before it's discovered from bad
notifications in production.

This is the baseline any future prompt change or model-tier decision
(`_docs/design.md` §6/§16) should be compared against -- a change that
measurably regresses decision or match agreement here is a real signal,
not noise; the relevance/importance numbers have more natural slack
given the above.
