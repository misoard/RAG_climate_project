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

**Status:** M0 — contracts & scaffold.
