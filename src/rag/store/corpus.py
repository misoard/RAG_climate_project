"""Build the searchable corpus once and cache it, so a run costs milliseconds not minutes.

Embedding 174 chunks on CPU takes ~25 seconds. That is fine once and wasteful on
every run, and M2's eval loop will start the pipeline repeatedly, so the vectors
are cached to ``data/processed/``.

The cache key is the whole recipe -- report, embedding model, chunk size, overlap
-- because a cache that survives a settings change is worse than no cache: it
would serve vectors from the old model or the old chunker while the code believes
it is testing the new one, and quietly corrupt every eval number that follows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contracts.models import Chunk
from rag.chunking.chunker import build_corpus
from rag.embeddings.embedder import Embedder
from rag.settings import load_settings
from rag.store.inmemory import InMemoryStore

_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = _ROOT / "data" / "processed"


def _cache_path(report: str) -> Path:
    """A filename that changes whenever anything upstream of the vectors changes."""
    settings = load_settings()
    model_slug = settings.embeddings.model.replace("/", "-")
    chunking = f"{settings.chunking.strategy}-{settings.chunking.chunk_size}-{settings.chunking.chunk_overlap}"
    return PROCESSED_DIR / f"{report}__{model_slug}__{chunking}.npz"


def build_store(report: str | None = None, *, refresh: bool = False) -> InMemoryStore:
    """Load the store from cache, or build and cache it. The corpus entry point."""
    settings = load_settings()
    report = report or settings.corpus.reports[0]
    cache = _cache_path(report)
    store = InMemoryStore(Embedder())

    if cache.exists() and not refresh:
        with np.load(cache, allow_pickle=False) as data:
            chunks = [Chunk.model_validate(c) for c in json.loads(str(data["chunks"]))]
            store.add_vectors(chunks, data["vectors"])
        return store

    chunks = build_corpus(report)
    store.add(chunks, show_progress=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        vectors=store.vectors,
        chunks=json.dumps([c.model_dump() for c in chunks]),
    )
    return store


if __name__ == "__main__":  # pragma: no cover - manual QA entry point
    import asyncio
    import time

    start = time.time()
    store = build_store(refresh="--refresh" in __import__("sys").argv)
    print(f"store of {len(store)} chunks ready in {time.time() - start:.1f}s")
    print(f"cache: {_cache_path(load_settings().corpus.reports[0]).name}")

    async def probe() -> None:
        for question in [
            "How much has global surface temperature risen?",
            "What is the best recipe for banana bread?",
        ]:
            hits = await store.retrieve(question, k=2)
            print(f"\n{question}")
            for h in hits:
                print(f"  {h.score:.3f}  {h.chunk.chunk_id} p{h.chunk.page_start}-{h.chunk.page_end}")

    asyncio.run(probe())
