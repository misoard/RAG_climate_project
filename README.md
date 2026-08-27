# IPCC Climate Q&A (RAG)

A grounded, cited question-answering tool over the IPCC AR6 corpus. It answers open-ended climate
questions **only** from AR6 material, cites report + section + page for every claim, preserves the
IPCC's own confidence and likelihood qualifiers ("high confidence", "very likely"), and refuses when
the corpus doesn't support an answer. It is deliberately *not* a source of local projections, policy
scorecards, or live data, and it never presents an IPCC assessment as an IPCC recommendation — the
IPCC is policy-relevant, not policy-prescriptive. The success test is that a journalist, teacher, or
student could safely quote an answer, with its citation and its uncertainty intact. This is a
learning project built milestone by milestone: see `CLAUDE.md` for how the work is run and
`MILESTONES.md` for the build sequence.

**Status:** M1 — walking skeleton. One `run_sequential` (retrieve → generate) runs end
to end over the AR6 SYR SPM: 174 naive fixed-size chunks, local multilingual `bge-m3`
embeddings, a brute-force in-memory cosine store, and a generation agent whose `Answer`
schema the `agentic-core` Gateway enforces. Retrieval quality is deliberately poor —
M2 adds the eval gate, and M3–M5 improve it against measured numbers.

## Running it

```bash
uv sync
cp .env.example .env          # then add your OPENROUTER_API_KEY
uv run python -m app.workflow "How much has global surface temperature risen?"
uv run pytest -q              # 21 tests, fully offline (no key, no network)
```

The first run downloads the SPM (~5MB) and the embedding model (~2.2GB) and takes
about 25 seconds to embed; vectors are then cached to `data/processed/` and later
runs start instantly.
