"""The in-memory retriever: ranking, filtering, and protocol conformance."""

from __future__ import annotations

import numpy as np
import pytest

from contracts.models import Retriever
from conftest import FakeEmbedder, make_chunk
from rag.store.inmemory import InMemoryStore


def test_satisfies_the_retriever_protocol(store):
    # Structural, not nominal: the store neither imports nor inherits Retriever.
    assert isinstance(store, Retriever)


async def test_ranks_the_relevant_chunk_first(store):
    hits = await store.retrieve("temperature warming", k=3)
    assert hits[0].chunk.chunk_id == "AR6-SYR#0001"
    # Scores must come back sorted best-first, since callers truncate by position.
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


async def test_k_caps_the_number_of_results(store):
    assert len(await store.retrieve("temperature", k=2)) == 2


async def test_k_larger_than_the_corpus_is_not_an_error(store):
    assert len(await store.retrieve("temperature", k=99)) == 3


async def test_filters_restrict_candidates(store):
    hits = await store.retrieve("ocean", k=5, filters={"working_group": "WG1"})
    assert [h.chunk.chunk_id for h in hits] == ["AR6-SYR#0003"]


async def test_filter_matching_nothing_returns_nothing(store):
    assert await store.retrieve("ocean", k=5, filters={"working_group": "WG3"}) == []


async def test_empty_store_returns_nothing():
    assert await InMemoryStore(FakeEmbedder()).retrieve("anything", k=3) == []


def test_add_vectors_rejects_a_length_mismatch():
    store = InMemoryStore(FakeEmbedder())
    with pytest.raises(ValueError, match="chunks but"):
        store.add_vectors([make_chunk("a", "x")], np.zeros((2, 6), dtype=np.float32))
