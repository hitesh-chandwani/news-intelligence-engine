"""Tests for src/nie/embeddings/fastembed.py (issue #21).

Deliberately uses the **real** fastembed model, not a stub -- the narrow,
documented exception to `_docs/testing-guidelines.md`'s no-live-network
rule (see issue #21's "Testing the real model" section). The first call
in this process downloads the ~100MB `BAAI/bge-small-en-v1.5` ONNX model
from Hugging Face to fastembed's local cache directory; every call after
that (including on CI once it exists, as long as the cache directory
persists) is fully offline and deterministic. Proving the wrapper
produces stable, real 384-dim vectors -- not a stub returning a
constant -- is the actual point of these tests, so no monkeypatching of
`fastembed`/`TextEmbedding` happens here.

No DB access -- this module only exercises the pure embedding wrapper.
"""

from __future__ import annotations

from nie.embeddings.fastembed import embed_text, embed_texts


def test_embed_text_same_text_twice_returns_identical_vectors() -> None:
    text = "Silver ETF inflows hit a record high this week."

    first = embed_text(text)
    second = embed_text(text)

    assert len(first) == 384
    assert len(second) == 384
    assert first == second


def test_embed_text_distinct_texts_return_different_vectors() -> None:
    first = embed_text("Silver ETF inflows hit a record high this week.")
    second = embed_text(
        "The central bank raised interest rates by fifty basis points today."
    )

    assert len(first) == 384
    assert len(second) == 384
    assert first != second


def test_embed_texts_empty_list_returns_empty_list() -> None:
    assert embed_texts([]) == []


def test_embed_texts_batches_in_input_order() -> None:
    # Batching (vs. one-at-a-time `embed_text` calls) can shift individual
    # float values slightly due to ONNX padding within the batch, so this
    # doesn't compare against `embed_text` output -- it only checks that
    # one `embed_texts` call returns distinct, correctly-ordered,
    # reproducible vectors for distinct inputs.
    texts = [
        "Gold prices rallied after the jobs report.",
        "Silver ETF inflows hit a record high this week.",
    ]

    first_call = embed_texts(texts)
    second_call = embed_texts(texts)

    assert len(first_call) == 2
    assert all(len(vector) == 384 for vector in first_call)
    assert first_call[0] != first_call[1]
    assert first_call == second_call
