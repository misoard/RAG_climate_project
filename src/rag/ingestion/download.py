"""Fetch AR6 source PDFs into ``data/raw/`` and record where they came from.

The corpus is the one input this project does not own, so acquiring it is treated
as a step with provenance rather than a manual drag-and-drop: every download is
recorded in ``data/manifest.json`` with its URL, size and SHA-256. That hash is
what lets a later milestone answer "was the eval baseline measured against this
exact file?" -- IPCC PDFs are occasionally re-issued at the same URL.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import urllib.request
from datetime import date
from pathlib import Path

import certifi

from pydantic import BaseModel

from contracts.models import DocumentLayer, WorkingGroup

# Repo root, from src/rag/ingestion/download.py -> up four levels.
_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = _ROOT / "data" / "raw"
MANIFEST = _ROOT / "data" / "manifest.json"


class Source(BaseModel):
    """One downloadable AR6 document, and the provenance a Chunk will inherit from it.

    ``report``/``working_group``/``document_layer`` are carried here rather than
    guessed at parse time: they are facts about *which document this is*, known
    when we choose to download it, and nothing in the PDF states them reliably.
    """

    report: str
    working_group: WorkingGroup
    document_layer: DocumentLayer
    url: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


# The catalogue of what this project can ingest. URLs are structural (they change
# only when the IPCC reorganises its site), so they live in code, not in .env.
# config/settings.yaml picks *which* of these M1 actually uses.
SOURCES: dict[str, Source] = {
    "AR6-SYR": Source(
        report="AR6-SYR",
        working_group="SYR",
        document_layer="SPM",
        url="https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf",
    ),
}


def _fetch(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest``, verifying TLS against certifi's CA bundle.

    The explicit context is not optional hardening: a python.org framework build
    ships no CA store of its own, so the stdlib default fails to verify ipcc.ch
    with CERTIFICATE_VERIFY_FAILED. Pointing at certifi fixes that *without* the
    usual copy-pasted "disable verification" workaround, which would leave us
    trusting whatever answers on the way to the corpus.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    # A UA: some CDNs reject the default "Python-urllib/3.x" outright.
    request = urllib.request.Request(url, headers={"User-Agent": "ipcc-rag/0.1 (+research)"})
    with urllib.request.urlopen(request, context=context) as response, dest.open("wb") as out:
        shutil.copyfileobj(response, out)  # streamed, so a large PDF never lands in memory


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        # Chunked so a 100MB WG1 report doesn't get read into memory whole (M8).
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _record(source: Source, path: Path) -> None:
    """Add or update this source's entry in the manifest.

    ``downloaded`` records when these *bytes* were obtained, not when the manifest
    was last written -- so a cache hit must not touch it. Re-running ingestion is
    a read, and a provenance field that silently advances on every read asserts
    something false, which is worse than not recording it at all. The date is
    therefore carried forward whenever the hash shows the file is unchanged, and
    only reset when the content actually differs (a re-issued PDF, or a manual
    replacement in data/raw/).
    """
    entries = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    digest = _sha256(path)
    previous = entries.get(source.report, {})
    obtained = (
        previous["downloaded"]
        if previous.get("sha256") == digest and "downloaded" in previous
        else date.today().isoformat()
    )
    entries[source.report] = {
        "report": source.report,
        "working_group": source.working_group,
        "document_layer": source.document_layer,
        "url": source.url,
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "downloaded": obtained,
    }
    MANIFEST.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


def download(source: Source, *, force: bool = False) -> Path:
    """Fetch ``source`` into data/raw/, skipping the network if we already have it.

    Idempotent by default: re-running ingestion should not re-download 40MB. Pass
    ``force=True`` to refresh a file the IPCC has re-issued.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / source.filename
    if not path.exists() or force:
        _fetch(source.url, path)
    _record(source, path)
    return path


def ensure(report: str, *, force: bool = False) -> Path:
    """Download by report id (``"AR6-SYR"``), the form settings.yaml uses."""
    if report not in SOURCES:
        raise KeyError(f"unknown report {report!r}; known: {sorted(SOURCES)}")
    return download(SOURCES[report], force=force)


if __name__ == "__main__":  # pragma: no cover - operational entry point
    for name in SOURCES:
        print(f"{name}: {ensure(name)}")
