"""Brute-force in-memory cosine store -- the M1 ``Retriever``, and no more than that.

It holds every vector in one numpy array and scores a query against all of them
with a single matrix multiply. At the scale of one SPM (174 chunks) that is not a
compromise, it is simply the right answer: an exact search over 174x1024 floats
takes well under a millisecond, and any vector database would add a dependency, a
process and an index-tuning problem to be *slower and approximate*.

It stops being the right answer somewhere in the tens of thousands of chunks,
which is what M5's first lever addresses. Because it satisfies the ``Retriever``
protocol, that swap changes this file and nothing else -- which is the entire
reason the protocol was fixed in M0 before any of this existed.
"""

from __future__ import annotations

import numpy as np

from contracts.models import Chunk, RetrievedChunk
from rag.embeddings.embedder import Embedder


class InMemoryStore:
    """An exact cosine-similarity store over a list of chunks.

    Satisfies ``contracts.models.Retriever`` structurally -- it neither imports nor
    inherits from it, which is the point of a Protocol.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def vectors(self) -> np.ndarray | None:
        """The stored matrix, for callers that persist it (see store/corpus.py)."""
        return self._vectors

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk], *, show_progress: bool = False) -> None:
        """Embed and store chunks. Re-embeds nothing that is already here."""
        if not chunks:
            return
        vectors = self.embedder.encode([c.text for c in chunks], show_progress=show_progress)
        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def add_vectors(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Store chunks with vectors computed elsewhere (a cache, or a test fixture).

        The seam that keeps tests offline: a test can hand over hand-made vectors and
        never load a 2.2GB model.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self._chunks.extend(chunks)
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    @staticmethod
    def _matches(chunk: Chunk, filters: dict) -> bool:
        """Exact-match metadata filter, e.g. ``{"working_group": "WG1"}``.

        Plain equality only. Deciding *which* filter a question implies (routing a
        sea-level question to WG1) is M5's job; this is just the mechanism it will
        use. Filters are applied rather than silently ignored, because a caller that
        passes one and gets unfiltered results back has a bug it cannot see.
        """
        return all(getattr(chunk, field, None) == value for field, value in filters.items())

    async def retrieve(
        self, query: str, k: int = 8, filters: dict | None = None
    ) -> list[RetrievedChunk]:
        """Return the k chunks most similar to ``query``, best first.

        ``async`` despite numpy being entirely synchronous: the protocol is async
        because the orchestration is, and because the retrievers that replace this
        one (a vector DB, an HTTP reranker) will genuinely await. Making the cheap
        implementation match the interface now means M5 changes no call sites.
        """
        if self._vectors is None or not self._chunks:
            return []

        candidates = range(len(self._chunks))
        if filters:
            candidates = [i for i in candidates if self._matches(self._chunks[i], filters)]
            if not candidates:
                return []

        # Vectors are unit-normalized at encode time, so this dot product IS cosine
        # similarity -- no norm division needed.
        query_vector = self.embedder.encode_one(query)
        index = np.asarray(candidates, dtype=np.int64)
        scores = self._vectors[index] @ query_vector

        # argpartition finds the top k without sorting all 174; then sort just those.
        k = min(k, len(index))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            RetrievedChunk(chunk=self._chunks[int(index[i])], score=float(scores[i])) for i in top
        ]
