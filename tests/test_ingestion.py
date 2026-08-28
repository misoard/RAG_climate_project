"""Manifest provenance: the download record must describe the bytes, not the run."""

from __future__ import annotations

import json
from datetime import date

from rag.ingestion import download as dl
from rag.ingestion.download import Source

SOURCE = Source(
    report="AR6-SYR",
    working_group="SYR",
    document_layer="SPM",
    url="https://example.invalid/spm.pdf",
)


def _manifest(tmp_path, monkeypatch, content: bytes = b"pdf-bytes"):
    """Point the module at a throwaway manifest and give it a file to record."""
    monkeypatch.setattr(dl, "MANIFEST", tmp_path / "manifest.json")
    path = tmp_path / "spm.pdf"
    path.write_bytes(content)
    return path


def test_record_captures_size_and_hash(tmp_path, monkeypatch):
    path = _manifest(tmp_path, monkeypatch)
    dl._record(SOURCE, path)

    entry = json.loads(dl.MANIFEST.read_text())["AR6-SYR"]
    assert entry["bytes"] == len(b"pdf-bytes")
    assert entry["sha256"] == dl._sha256(path)
    assert entry["url"] == SOURCE.url


def test_rerecording_unchanged_bytes_preserves_the_original_date(tmp_path, monkeypatch):
    """A cache hit is a read. It must not advance the provenance date.

    Re-running ingestion re-records the entry; if that reset ``downloaded`` to today,
    the manifest would claim the corpus was fetched on a day it wasn't -- and from M2
    on, the manifest is how we say which bytes a baseline was measured against.
    """
    path = _manifest(tmp_path, monkeypatch)
    dl._record(SOURCE, path)

    # Backdate the entry, then re-record the very same file.
    entries = json.loads(dl.MANIFEST.read_text())
    entries["AR6-SYR"]["downloaded"] = "2020-01-01"
    dl.MANIFEST.write_text(json.dumps(entries))
    dl._record(SOURCE, path)

    assert json.loads(dl.MANIFEST.read_text())["AR6-SYR"]["downloaded"] == "2020-01-01"


def test_changed_bytes_reset_the_date(tmp_path, monkeypatch):
    """A re-issued PDF is genuinely new provenance, so the date must move."""
    path = _manifest(tmp_path, monkeypatch)
    dl._record(SOURCE, path)
    entries = json.loads(dl.MANIFEST.read_text())
    entries["AR6-SYR"]["downloaded"] = "2020-01-01"
    dl.MANIFEST.write_text(json.dumps(entries))

    path.write_bytes(b"a re-issued edition")  # same URL, different content
    dl._record(SOURCE, path)

    entry = json.loads(dl.MANIFEST.read_text())["AR6-SYR"]
    assert entry["downloaded"] == date.today().isoformat()
    assert entry["sha256"] == dl._sha256(path)
