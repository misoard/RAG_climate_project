"""Typed loader for ``config/settings.yaml`` -- the retrieval half's tunable knobs.

Named ``RagSettings`` rather than ``Settings`` on purpose: ``agentic_core.Settings``
already exists and means something different (env/secrets for the model call path).
The two do not overlap. This file is the values that get swept during the eval
milestones; that one is the values that vary per environment. Secrets live in
neither -- they come from ``.env``.

Validated with Pydantic like every other boundary: a typo in the YAML should fail
at load naming the field, not surface as a ``KeyError`` deep in the chunker.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = _ROOT / "config" / "settings.yaml"


class CorpusSettings(BaseModel):
    reports: list[str] = Field(default_factory=list)
    language: str = "en"


class ChunkingSettings(BaseModel):
    strategy: Literal["fixed", "section_aware"] = "fixed"
    chunk_size: int = 1000
    chunk_overlap: int = 150


class EmbeddingSettings(BaseModel):
    # No default: the model fixes the vector space, so an accidentally-defaulted
    # value would silently produce a store that cannot be searched by the model
    # everything else assumes.
    model: str
    normalize: bool = True
    batch_size: int = 32


class RetrievalSettings(BaseModel):
    top_k: int = 8
    candidate_k: int = 40
    rerank: bool = False
    hybrid: bool = False


class ThresholdSettings(BaseModel):
    # None until calibrated in M6. Score scales are retriever-specific, so there is
    # no honest default to put here.
    min_score_to_answer: float | None = None
    min_chunks_to_answer: int = 1


class ModelAliases(BaseModel):
    generation: str = "smart"
    judge: str = "fast"


class RagSettings(BaseModel):
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embeddings: EmbeddingSettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    models: ModelAliases = Field(default_factory=ModelAliases)


@lru_cache(maxsize=1)
def load_settings(path: Path | None = None) -> RagSettings:
    """Load and validate settings.yaml. Cached -- it is read on every retrieval call."""
    return RagSettings.model_validate(yaml.safe_load((path or SETTINGS_PATH).read_text()))
