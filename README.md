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
