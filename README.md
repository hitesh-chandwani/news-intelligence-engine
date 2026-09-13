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
