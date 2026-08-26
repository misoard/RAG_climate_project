"""The stable boundary types (CLAUDE.md 6).

These are the *contracts*: the only types that cross the line between the retrieval
half (our code, in ``rag/``) and the generation half (agentic-core Agents in ``app/``).
Everything on either side is a swappable implementation. Fixing these now is what lets
the naive M1 in-memory store be replaced by a real vector DB in M5, or the generation
prompt be rewritten in M6, without either side knowing.

Changing anything here is a deliberate act, not a side effect of another change.
"""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# Which AR6 (and cycle-adjacent special report) product a chunk came from.
WorkingGroup = Literal["WG1", "WG2", "WG3", "SYR", "SR1.5", "SROCC", "SRCCL"]

# Where in a report the text sits. This matters for trust and for routing: an SPM
# headline is negotiated, approved language; a chapter paragraph is the underlying
# assessment; an FAQ is the authors' own plain-language gloss (and the source of the
# M2 gold set).
DocumentLayer = Literal["SPM", "TS", "chapter", "FAQ"]


# ---- the unit of retrieval ----


class Chunk(BaseModel):
    """One retrievable passage plus everything needed to cite and qualify it.

    The metadata is not decoration. ``page``/``section``/``report`` are what make an
    answer checkable; the ``*_terms`` lists are what let us detect (M6) that an answer
    dropped the IPCC's own hedging and stated a "likely" range as a certainty.
    """

    chunk_id: str
    text: str

    # --- provenance: enough to cite precisely ---
    report: str  # "AR6-SYR", "AR6-WG1", ...
    working_group: WorkingGroup
    document_layer: DocumentLayer
    chapter: str | None = None
    section: str | None = None
    page: int

    # --- assessment signals: populated properly in M3 ---
    # A headline statement is the report's own top-level claim; worth knowing because
    # it is usually the best possible answer to a broad question.
    is_headline_statement: bool = False
    confidence_terms: list[str] = Field(default_factory=list)  # e.g. ["high confidence"]
    likelihood_terms: list[str] = Field(default_factory=list)  # e.g. ["very likely"]
    scenarios_mentioned: list[str] = Field(default_factory=list)  # e.g. ["SSP1-2.6"]
    figure_or_table_ref: str | None = None


class RetrievedChunk(BaseModel):
    """A chunk together with the score the retriever gave it for a given query.

    The score is deliberately untyped as to its scale (cosine, BM25, cross-encoder):
    only the retriever that produced it may interpret it, and the refusal threshold in
    M6 is calibrated per retriever rather than assumed to be comparable across them.
    """

    chunk: Chunk
    score: float


# ---- the retrieval boundary ----


@runtime_checkable
class Retriever(Protocol):
    """What the generation half is allowed to assume about retrieval: this and nothing else.

    A Protocol (structural typing) rather than a base class, so implementations don't
    have to import or inherit from us -- they just have to have this method.
    """

    async def retrieve(
        self, query: str, k: int = 8, filters: dict | None = None
    ) -> list[RetrievedChunk]: ...


# ---- the generation boundary (the Agent's typed I/O) ----


class Citation(BaseModel):
    """A pointer a reader can follow back into the PDF."""

    report: str
    document_layer: str
    section: str | None = None
    page: int


class GenerationInput(BaseModel):
    """Everything the generation Agent is given -- and, by omission, all it may use.

    ``allowed_chunk_ids`` exists so a citation can be checked *after* generation against
    what was actually retrieved: a model that cites a chunk it was never shown has
    fabricated it, and we can prove that mechanically rather than by judgement.
    """

    question: str
    context_block: str  # rendered from retrieved chunks, with inline source markers
    allowed_chunk_ids: list[str]  # for post-hoc citation validation


class Answer(BaseModel):
    """The Agent's typed output -- the schema the Gateway's re-prompt loop enforces.

    Making refusal a *field* rather than an absence of text is the point: "the corpus
    does not support this" is a first-class, evaluable outcome (M2 refusal correctness),
    not a failure mode.
    """

    text: str
    citations: list[Citation] = Field(default_factory=list)
    # Did the answer keep the IPCC's confidence/likelihood language where the source had it?
    qualifiers_preserved: bool
    # True when the corpus does not support an answer.
    refused: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)


__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "DocumentLayer",
    "GenerationInput",
    "RetrievedChunk",
    "Retriever",
    "WorkingGroup",
]
