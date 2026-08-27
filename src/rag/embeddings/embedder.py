"""Local sentence-transformers embeddings. Free, private, and multilingual by design.

The model is fixed in ``config/settings.yaml`` (BAAI/bge-m3) and is the one choice
in this project that is expensive to reverse: it defines the vector space, so
changing it invalidates every stored vector and means re-embedding the corpus.
Multilingual is therefore not a nice-to-have but the requirement -- a French
question has to land near English text in this same space for the planned French
extension to work at the edges instead of forcing a second store.

bge-m3 needs no "query:"/"passage:" prefixes (unlike the E5 family), so queries
and documents go through the same call. That removes a whole class of silent bug
where the two sides get encoded asymmetrically and retrieval quietly degrades.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.settings import load_settings


@lru_cache(maxsize=2)
def _load_model(name: str) -> SentenceTransformer:
    """Load (and cache) the model. ~2.2GB, so loading it twice is a real cost."""
    return SentenceTransformer(name)


class Embedder:
    """Turns text into vectors, with the store's normalization convention applied.

    ``normalize`` is on by default and matters more than it looks: with unit-length
    vectors, cosine similarity *is* the dot product, so the in-memory store can rank
    with a single matrix multiply and no per-query norm division.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        normalize: bool | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = load_settings().embeddings
        self.model_name = model_name or settings.model
        self.normalize = settings.normalize if normalize is None else normalize
        self.batch_size = batch_size or settings.batch_size
        # Deliberately not loaded here: constructing an Embedder must stay cheap so
        # importing this module doesn't drag 2.2GB into memory during tests.
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_embedding_dimension())

    def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        """Encode a batch of texts to a (len(texts), dim) float32 array."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single query to a (dim,) vector."""
        return self.encode([text])[0]


if __name__ == "__main__":  # pragma: no cover - manual QA entry point
    embedder = Embedder()
    probe = embedder.encode(["global surface temperature has increased", "banana bread recipe"])
    query = embedder.encode_one("how much has the world warmed?")
    print(f"model {embedder.model_name}  dim {embedder.dimension}")
    print(f"norms {np.linalg.norm(probe, axis=1).round(4)}  (1.0 => normalized)")
    print(f"similarity to warming sentence : {float(query @ probe[0]):.4f}")
    print(f"similarity to banana bread     : {float(query @ probe[1]):.4f}")

    # The cross-lingual check: this is the property bge-m3 was chosen for, so it is
    # worth asserting rather than assuming. A French question must land nearer the
    # relevant English sentence than an irrelevant one, in this same vector space.
    fr = embedder.encode_one("de combien la planete s'est-elle rechauffee ?")
    print(f"FR query -> EN warming sentence : {float(fr @ probe[0]):.4f}")
    print(f"FR query -> EN banana bread     : {float(fr @ probe[1]):.4f}")
