"""Crude PDF -> text extraction. The M1 parse, deliberately the dumbest that works.

M1's job is to prove the plumbing, not to parse well, so this does exactly two
things: pull each page's text as PyMuPDF hands it over, and work out the page
number a reader would actually cite. Everything a real parse owes the corpus --
stripping running headers, joining the two-column layout in reading order,
keeping figure captions with their figures, recovering section headings -- is
M3, and is left visibly undone here so M3's eval delta measures something real.

The one thing not left to M3 is the page number, because M1 must emit citations
and a citation to the wrong page is not a crude citation, it is a wrong one.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from pydantic import BaseModel


class Page(BaseModel):
    """One page of extracted text, with both ways of naming the page.

    The two numbers differ and the difference matters. ``pdf_index`` is where the
    page sits in the file; ``page_number`` is what is printed on it -- in this SPM
    they are six apart, because front matter is numbered separately. Citations must
    use ``page_number`` (it is what a reader sees), while debugging and manual QA
    want ``pdf_index`` (it is what a PDF viewer's page box takes).
    """

    pdf_index: int  # 1-based position in the file
    page_number: int  # the number printed on the page, for citation
    numbering_is_printed: bool  # False => page_number fell back to pdf_index
    text: str


def _printed_page_number(lines: list[str]) -> int | None:
    """Read the page number out of the running header, if it is there.

    In this SPM the header extracts as its own leading line holding nothing but
    the digits (the "| Summary for Policymakers" beside it comes through as a
    separate line). Front matter is numbered with roman numerals, which this
    deliberately does not match -- returning None there is correct, not a miss.
    """
    if lines and lines[0].strip().isdigit():
        return int(lines[0].strip())
    return None


def parse_pdf(path: Path | str) -> list[Page]:
    """Extract every non-empty page of ``path`` in document order.

    Pages that yield no text at all (covers and full-bleed figure pages, of which
    this SPM has five) are dropped rather than carried as empty strings: they
    cannot be retrieved or cited, and keeping them would only add empty chunks.
    """
    pages: list[Page] = []
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text.strip():
                continue
            printed = _printed_page_number(text.splitlines())
            pages.append(
                Page(
                    pdf_index=index,
                    page_number=printed if printed is not None else index,
                    numbering_is_printed=printed is not None,
                    # Text kept exactly as extracted, running header and all. Cleaning
                    # it is M3's measurable win; doing it here would hide that.
                    text=text,
                )
            )
    return pages


if __name__ == "__main__":  # pragma: no cover - manual QA entry point
    import sys

    from rag.ingestion.download import ensure

    pages = parse_pdf(ensure(sys.argv[1] if len(sys.argv) > 1 else "AR6-SYR"))
    printed = sum(p.numbering_is_printed for p in pages)
    print(f"{len(pages)} pages with text; {printed} carry a printed page number")
    for p in pages[:3] + pages[-2:]:
        flag = "" if p.numbering_is_printed else "  (fallback)"
        print(f"  pdf {p.pdf_index:>2} -> page {p.page_number:>3}{flag}  {len(p.text):>5} chars")
