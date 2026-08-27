"""The naive chunker: does it cut where it says, and are page spans honest?"""

from __future__ import annotations

import pytest

from rag.chunking.chunker import chunk_pages
from rag.ingestion.download import Source
from rag.ingestion.parse import Page

SOURCE = Source(
    report="AR6-SYR",
    working_group="SYR",
    document_layer="SPM",
    url="https://example.invalid/spm.pdf",
)


def pages(*texts: str) -> list[Page]:
    return [
        Page(pdf_index=i, page_number=i, numbering_is_printed=True, text=t)
        for i, t in enumerate(texts, start=1)
    ]


def test_chunks_respect_size_and_overlap():
    chunks = chunk_pages(pages("A" * 250), SOURCE, chunk_size=100, chunk_overlap=20)
    assert all(len(c.text) <= 100 for c in chunks)
    # Window advances by size - overlap, so consecutive chunks share `overlap` chars.
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_chunk_spanning_a_page_break_records_both_pages():
    # One chunk larger than either page must cover both.
    chunks = chunk_pages(pages("A" * 60, "B" * 60), SOURCE, chunk_size=200, chunk_overlap=0)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_chunk_inside_one_page_collapses_to_a_single_page():
    chunks = chunk_pages(pages("A" * 500), SOURCE, chunk_size=100, chunk_overlap=0)
    assert all(c.page_start == c.page_end == 1 for c in chunks)


def test_chunks_inherit_provenance_from_the_source():
    chunk = chunk_pages(pages("some text"), SOURCE, chunk_size=100, chunk_overlap=0)[0]
    assert (chunk.report, chunk.working_group, chunk.document_layer) == ("AR6-SYR", "SYR", "SPM")
    # M3 populates these; M1 must leave them empty rather than guess.
    assert chunk.confidence_terms == [] and chunk.section is None


def test_overlap_at_least_size_is_rejected():
    # Would otherwise never advance the window and loop forever.
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_pages(pages("text"), SOURCE, chunk_size=100, chunk_overlap=100)


def test_no_pages_gives_no_chunks():
    assert chunk_pages([], SOURCE, chunk_size=100, chunk_overlap=10) == []
