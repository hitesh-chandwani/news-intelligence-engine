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

**PASS BAR: not yet established.** This harness has not yet been run
against a real, configured `LLM_API_KEY` in any environment -- the
agreement percentages it reports on a first real run need to be recorded
here (decision agreement, existing-event match agreement, relevance
exact-match, importance exact-match) before this can gate anything. Run
the command above with real credentials and replace this paragraph with
the actual numbers it reports.
