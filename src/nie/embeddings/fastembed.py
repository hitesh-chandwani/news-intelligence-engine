"""Local embeddings wrapper around FastEmbed (#21, `design.md` §3, §14).

Exposes `embed_texts`/`embed_text`, both backed by a single, lazily
constructed, module-level `fastembed.TextEmbedding` instance rather than
the original one-function `embed(text) -> list[float]` sketch. A
`TextEmbedding` instance is expensive to build -- constructing it loads
an ONNX model into memory -- but encoding after that is cheap (~5ms/doc
per `design.md` §3). This module runs inside a long-running process
(`design.md` §2: one process, a scheduler firing the pipeline
repeatedly), so building the model once per process and reusing it
across every pipeline run avoids reloading it on every call. `embed_texts`
takes a batch so `nie.pipeline.embed.embed_stage` can embed every row it
selects in one call per pipeline run instead of looping row-by-row
through `embed_text` and re-paying per-call overhead.

The model name is read from `Settings().embedding_model` (#3,
`EMBEDDING_MODEL`, default `BAAI/bge-small-en-v1.5`) each time the
singleton is built -- never hardcoded here.
"""

from __future__ import annotations

import fastembed

from nie.config import Settings

_model: fastembed.TextEmbedding | None = None


def _get_model() -> fastembed.TextEmbedding:
    """Return the process-wide `TextEmbedding` singleton, building it on
    first use.

    Reads `Settings().embedding_model` at construction time, not import
    time, so tests/config overrides via environment variables still take
    effect as long as they're set before the first embedding call.
    """
    global _model
    if _model is None:
        _model = fastembed.TextEmbedding(model_name=Settings().embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings in one call, in input order.

    Returns one 384-length `list[float]` per input string, converted from
    fastembed's `numpy.ndarray` output -- plain lists are what
    `Source.embedding: Mapped[list[float] | None]`/pgvector expect.

    An empty `texts` list returns `[]` immediately without calling into
    fastembed (and without building the model singleton if it hasn't been
    built yet).
    """
    if not texts:
        return []
    model = _get_model()
    return [vector.tolist() for vector in model.embed(texts)]


def embed_text(text: str) -> list[float]:
    """Embed a single string. Convenience wrapper: `embed_texts([text])[0]`."""
    return embed_texts([text])[0]
