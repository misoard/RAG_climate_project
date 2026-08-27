"""Shared offline fixtures. No network, no API key, no 2.2GB model.

The two things that make the real pipeline expensive -- the embedding model and the
LLM -- are both replaced here at their seams: ``FakeEmbedder`` for the first,
``FakeRouter`` for the second. Everything between them is the real code path.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from contracts.models import Chunk

# A vocabulary small enough to reason about: each text becomes a normalized
# bag-of-words vector over these terms, so "similarity" in tests is predictable
# rather than a black box.
VOCAB = ["temperature", "warming", "emissions", "adaptation", "ocean", "bread"]


class FakeEmbedder:
    """Deterministic stand-in for the sentence-transformers embedder.

    Same interface as the real ``Embedder`` (encode / encode_one, unit-normalized),
    so the store under test is exercised exactly as in production -- only the vectors
    are cheap and predictable.
    """

    normalize = True
    batch_size = 8
    model_name = "fake-bow"

    @property
    def dimension(self) -> int:
        return len(VOCAB)

    def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            row = np.array([float(lowered.count(term)) for term in VOCAB], dtype=np.float32)
            norm = np.linalg.norm(row)
            rows.append(row / norm if norm else row)
        return np.vstack(rows).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def make_chunk(chunk_id: str, text: str, **overrides) -> Chunk:
    """A Chunk with sane defaults, so tests state only what they care about."""
    fields = {
        "chunk_id": chunk_id,
        "text": text,
        "report": "AR6-SYR",
        "working_group": "SYR",
        "document_layer": "SPM",
        "page_start": 1,
        "page_end": 1,
    }
    fields.update(overrides)
    return Chunk(**fields)


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk("AR6-SYR#0001", "Global surface temperature warming reached 1.1C.", page_start=4, page_end=4),
        make_chunk("AR6-SYR#0002", "Greenhouse gas emissions continued to rise.", page_start=5, page_end=6),
        make_chunk("AR6-SYR#0003", "Ocean heat content increased.", page_start=7, page_end=7, working_group="WG1"),
    ]


@pytest.fixture
def store(chunks):
    from rag.store.inmemory import InMemoryStore

    s = InMemoryStore(FakeEmbedder())
    s.add(chunks)
    return s


def answer_json(
    *,
    text: str = "Global surface temperature has warmed (high confidence).",
    citations: list[dict] | None = None,
    qualifiers_preserved: bool = True,
    refused: bool = False,
    supporting: list[str] | None = None,
) -> str:
    """A well-formed Answer as the model would return it."""
    if citations is None:
        citations = [
            {
                "chunk_id": "AR6-SYR#0001",
                "report": "AR6-SYR",
                "document_layer": "SPM",
                "section": "A.1.2",
                "page": 4,
            }
        ]
    return json.dumps(
        {
            "text": text,
            "citations": citations,
            "qualifiers_preserved": qualifiers_preserved,
            "refused": refused,
            "supporting_chunk_ids": supporting if supporting is not None else ["AR6-SYR#0001"],
        }
    )
