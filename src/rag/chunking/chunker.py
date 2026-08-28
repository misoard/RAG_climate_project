"""Naive fixed-size chunking with overlap -- the M1 splitter, and a bad one on purpose.

It cuts on a blind character count: mid-sentence, mid-table, and quite happily
between a claim and the "(high confidence)" that qualifies it. That last one is
the specifically dangerous failure for this corpus, and fixing it is M4's whole
job. Leaving it broken here is what gives M4 a real number to move.

The overlap is the only concession: a claim severed by one cut usually survives
whole in the neighbouring chunk.

One thing it does do properly is track page spans. Chunks are cut from the
concatenated document rather than page by page, so a chunk straddling a page
break is normal -- which is exactly why ``Chunk`` carries ``page_start``/
``page_end`` rather than a single page.
"""

from __future__ import annotations

from contracts.models import Chunk
from rag.ingestion.download import Source
from rag.ingestion.parse import Page

# Pages are joined with a blank line so the last word of one page and the first
# word of the next don't fuse into a nonexistent token.
_PAGE_SEPARATOR = "\n\n"


def _page_spans(pages: list[Page]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate pages, returning the text and each page's [start, end) offsets."""
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for page in pages:
        start = cursor
        parts.append(page.text)
        cursor += len(page.text)
        spans.append((start, cursor, page.page_number))
        parts.append(_PAGE_SEPARATOR)
        cursor += len(_PAGE_SEPARATOR)
    return "".join(parts), spans


def chunk_pages(
    pages: list[Page],
    source: Source,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split parsed pages into overlapping fixed-size chunks.

    ``source`` supplies the provenance every chunk inherits (report, working group,
    document layer): facts about which document this is, known at download time and
    not reliably recoverable from the text.
    """
    if chunk_overlap >= chunk_size:
        # Otherwise the window never advances and this loops forever.
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
    if not pages:
        return []

    text, spans = _page_spans(pages)
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        body = text[start:end]
        if body.strip():
            # Every page this window touches, so the citation range is honest even
            # when a chunk spans a break.
            touched = [num for (s, e, num) in spans if s < end and e > start]
            chunks.append(
                Chunk(
                    # Ordinal rather than a content hash: ids stay readable and sort
                    # into document order, which makes eval failures much easier to
                    # eyeball. The cost is that re-chunking renumbers everything.
                    #
                    # So: a chunk_id is valid ONLY within one build. Never persist one
                    # anywhere that outlives the build -- above all not in the eval gold
                    # set (see MILESTONES.md M2). M3 re-parses and M4 re-chunks, and a
                    # stored id would not fail loudly afterwards; it would silently point
                    # at different text and quietly corrupt every number measured with
                    # it. Reference chunks by a quoted span instead.
                    #
                    # A content hash would not rescue this: different chunk text hashes
                    # differently, so the id still moves. The instability is inherent to
                    # re-chunking, not to the id scheme.
                    chunk_id=f"{source.report}#{len(chunks):04d}",
                    text=body,
                    report=source.report,
                    working_group=source.working_group,
                    document_layer=source.document_layer,
                    page_start=min(touched),
                    page_end=max(touched),
                    # chapter/section and the confidence, likelihood and scenario
                    # terms stay empty here: detecting them is M3's work.
                )
            )
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def build_corpus(report: str) -> list[Chunk]:
    """Download, parse and chunk one report end to end, using settings.yaml's knobs."""
    from rag.ingestion.download import SOURCES, ensure
    from rag.ingestion.parse import parse_pdf
    from rag.settings import load_settings

    settings = load_settings()
    pages = parse_pdf(ensure(report))
    return chunk_pages(
        pages,
        SOURCES[report],
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )


if __name__ == "__main__":  # pragma: no cover - manual QA entry point
    import sys

    chunks = build_corpus(sys.argv[1] if len(sys.argv) > 1 else "AR6-SYR")
    spanning = sum(c.page_start != c.page_end for c in chunks)
    print(f"{len(chunks)} chunks; {spanning} span a page break")
    for c in chunks[:2]:
        print(f"\n--- {c.chunk_id}  pages {c.page_start}-{c.page_end} ---")
        print(c.text[:200].replace("\n", " "))
