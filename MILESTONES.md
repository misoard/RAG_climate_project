# MILESTONES.md — IPCC RAG build sequence

Work these **in order**, one at a time. At the end of each, run the **Milestone Review** from `CLAUDE.md §0` and **stop until the human says continue**. The shape is: contracts → walking skeleton → measurement gate → deepen each component behind stable interfaces, in risk/value order.

Each milestone lists: **Goal**, **Build**, **Tools**, **Done when**, and **Stop & teach** (what to walk the human through at the gate).

---

## M0 — Contracts & scaffold
**Goal:** pin down the interfaces the two halves meet at, and a bare project skeleton. No logic yet.

**Build:**
- `uv` project; add `agentic-core` as a git dependency; `.env.example` with `OPENROUTER_API_KEY`.
- `src/contracts/models.py` with the types from `CLAUDE.md §6` (`Chunk`, `RetrievedChunk`, `Retriever`, `Citation`, `GenerationInput`, `Answer`).
- Empty package skeleton for `rag/` and `app/` (mirroring the core's `how_to_start` split), plus `config/settings.yaml` with placeholders (top-k, thresholds, model alias).
- **Read `agentic-core`'s source + `how_to_start/`** and write a short `NOTES.md` recording the *actual* signatures for `Gateway`, `Agent`, `run_sequential`, the eval harness, and `FakeRouter`.

**Tools:** `uv`, `pydantic`, `agentic-core`.

**Done when:** the contracts import cleanly, the skeleton exists, and `NOTES.md` records the confirmed core APIs.

**Stop & teach:** why contracts-first — how fixing these interfaces lets the retrieval and generation halves evolve independently. Show the `Answer` schema and explain that it's what the Gateway will enforce.

---

## M1 — Walking skeleton (thin slice, end to end, on the real architecture)
**Goal:** one bad-but-alive answer flowing all the way through. Prove the plumbing.

**Build:**
- Ingest **one SPM** with the crudest possible parse (plain text + page numbers). No metadata richness yet.
- **Naive fixed-size chunking** (with overlap).
- Local **multilingual** embeddings (a multilingual BGE/E5 variant — this locks the embedding space, so choosing it now avoids a full re-embed when French support is added later); a **brute-force in-memory cosine store** implementing the `Retriever` protocol (`numpy`, no DB).
- `app/config.py` → `build_gateway()` with a one-model registry.
- A retrieval **async step** that fills `State` and renders the `context_block`.
- One **generation `Agent`** (`GenerationInput` → `Answer`) through the `Gateway`, with a first grounding+citation prompt.
- `app/workflow.py`: a single `run_sequential` — retrieve → generate — returning an `Answer`.

**Tools:** `PyMuPDF`, `sentence-transformers`, `numpy`, `agentic-core`.

**Done when:** `uv run python -m app.workflow` answers a question about that SPM with at least one citation, and the Gateway rejects+re-prompts a malformed answer at least once (verify this happened).

**Stop & teach:** trace one request end to end. Explain how the retrieval step feeds the Agent, and how the Gateway's re-prompt loop is already enforcing the citation schema. Be explicit that retrieval quality is poor on purpose — that's the next several milestones.

---

## M2 — Evaluation gate (measure before optimizing)
**Goal:** you can score retrieval and answer quality, so every later change is justified by numbers. This comes *before* any optimization.

**Build:**
- `eval/build_gold_set.py`: start with ~15–25 items — a few hand-written Q/A with known source locations, a couple derived from AR6 chapter FAQs, and **at least three unanswerable questions** (to test refusal). Store as `data/eval/gold_set.jsonl`.
- `eval/run_eval.py`: drive the **core's `eval/harness.py`**. Metrics: (a) retrieval hit rate, (b) faithfulness, (c) refusal correctness, (d) uncertainty preservation.
- Record the **baseline** on the M1 skeleton.

**Tools:** `agentic-core` eval harness; `FakeRouter` for any offline pieces.

**Done when:** `uv run python -m eval.run_eval` prints a scored report you trust, with a saved baseline to compare against.

**Stop & teach:** what each metric means and why this gate exists *now* rather than later. Show the baseline and predict which metric each upcoming milestone should move.

---

## M3 — Real ingestion & metadata
**Goal:** replace the crude parse with quality structured extraction — the highest-leverage work in the whole project.

**Build:**
- Proper layout-aware parsing: correct page numbers, section headings, dropped headers/footers, preserved figure/table captions.
- Populate the full `Chunk` metadata: `working_group`, `document_layer`, `section`, `is_headline_statement`, and detected `confidence_terms` / `likelihood_terms` / `scenarios_mentioned`.
- **Manual QA**: dump a sample and eyeball it against the PDF.
- Re-run eval.

**Tools:** `PyMuPDF` and/or `docling`/`unstructured`.

**Done when:** any section can be dumped with correct page/section/WG and detected qualifiers; eval is re-run and the delta recorded.

**Stop & teach:** the parsing pitfalls of two-column scientific PDFs, and why "just chunk the PDF" fails. Show the eval delta from better ingestion alone.

---

## M4 — Chunking strategy
**Goal:** section-aware, qualifier-preserving chunking to replace the naive splitter.

**Build:** chunk on section boundaries; never sever a claim from its confidence/likelihood qualifier; tune size/overlap. Swap it in behind the existing interface. Re-run eval.

**Tools:** your code.

**Done when:** the new chunker is in place and its eval impact is measured against M3.

**Stop & teach:** the chunking trade-offs (context vs precision; why qualifier-splitting is uniquely dangerous here), with numbers.

---

## M5 — Retrieval quality (one lever at a time)
**Goal:** move the retrieval hit-rate number up, provably, one change at a time.

**Build (each validated against eval before keeping):**
1. Swap the brute-force store for a real vector store (`pgvector` or `Qdrant`) behind the `Retriever` interface — expect no quality change, just scale.
2. Add a **cross-encoder reranker**.
3. Add **hybrid search** (dense + BM25) — helps with scenario codes like `SSP5-8.5`.
4. Add **metadata filtering / working-group routing**.
5. Add **scenario/term normalization** ("worst case" → `SSP5-8.5`).

**Tools:** `pgvector`/`Qdrant`, `sentence-transformers` cross-encoder, a BM25 lib.

**Done when:** retrieval hit rate is meaningfully above the M2 baseline **and each lever's individual contribution is quantified**.

**Stop & teach:** after *each* sub-step, report its measured contribution. Some may not help on this corpus — keeping only what the numbers justify is the lesson.

---

## M6 — Faithfulness & refusal
**Goal:** answers you'd trust enough to quote.

**Build:**
- An **uncertainty-preservation check** (LLM-judge `Agent`, or heuristic) that flags dropped qualifiers.
- Refusal threshold on retrieval score; distinguish "science doesn't **support** this" from "science doesn't **address** this."
- Ensure citations are precise and verifiable. Re-run eval.

**Tools:** `agentic-core` (judge agent), guardrails for output policy.

**Done when:** faithfulness and uncertainty-preservation clear a bar you set (e.g. ≥95% of sampled answers preserve qualifiers), and refusal correctness is solid on the unanswerable set.

**Stop & teach:** how grounding verification works, and why the honest-representation rules (assessed ≠ recommended; support vs address) matter for this corpus specifically.

> **Deferred — French extension (do not build now):** generating the answer in the user's language while citing the English source is a generation-side concern that lands here when picked up. Keep it out of the core path for now; the multilingual embedding model chosen in M1 is the only enabling piece that had to exist early.

---

## M7 — Interface & deployment
Two distinct lessons under one milestone: first make it usable, then make it run as a deployed service. **Each part has its own Milestone Review and stop gate** — do not roll them together.

### M7A — Interface
**Goal:** something a non-technical person can use.

**Build:** a thin UI (FastAPI + minimal front, or Streamlit) showing the answer, its confidence framing, the citations with source links, and (optionally) the retrieved passages for transparency. Graceful refusals.

**Tools:** FastAPI or Streamlit.

**Done when:** a friend can get a trustworthy, cited answer without your help (still running locally).

**Stop & teach:** presenting grounded AI honestly — surfacing sources and uncertainty rather than hiding them. **Then stop.**

> **Deferred — French extension (do not build now):** accepting a French question and presenting a French UI belongs here when picked up. Retrieval stays English (cross-lingual against the English corpus); only input capture, optional query translation, and answer language are French-facing. Deferred so the core path stays single-language until it's solid.

### M7B — Deployment
**Goal:** run the serving app online, reading a prebuilt index from persistent storage that outlives the process. The offline/online split becomes physical.

**Build:**
- **Split the two lifecycles into two entry points:** an **ingestion job** (batch, run occasionally) that builds the index and writes it to object/blob storage under a *versioned key prefix* (e.g. `ar6/v1/` — chunks as JSONL, embeddings as `.npy`/parquet, plus a manifest); and the **serving app** (long-running) that, on startup, downloads that artifact and loads it into memory.
- **A blob-backed `Retriever` implementation** behind the existing contract: load the prebuilt index from storage at startup, serve brute-force cosine from RAM. The generation half doesn't change.
- **Config-driven index pointer:** the app reads which prefix to load from config, so publishing `ar6/v2/` and flipping the pointer is the whole deploy; flipping back is rollback. Never overwrite an existing prefix.
- **Containerize the serving app**; keep it stateless and disposable (source of truth is the blob store, not the container's RAM). Add a readiness check that only passes once the index is loaded.
- **Secrets & locality:** auth to storage via a managed identity / IAM role (not raw keys where avoidable); keep the bucket in the app's region.
- **Operational baseline:** wire the core's `configure()` observability (OTel spans over retrieval + generation) and an optional query/embedding cache.

**Tools:** object/blob storage (Azure Blob, S3, or GCS), a container host, `agentic-core` `configure()`.

**Done when:** the app runs as a container, loads the prebuilt index from blob storage at startup, serves cited answers over the network, and a new index version can be published and pointed to via config **with no code change**.

**Stop & teach:** why the serving container is stateless and disposable; where the vectors actually live (blob storage as source of truth vs RAM as working copy); versioned artifacts + pointer-flip deploys/rollbacks; and how the blob-backed retriever is just one more implementation behind the `Retriever` contract — invisible to generation. Note explicitly that blob storage *persists* the index but does **not** do the search; the app does, after loading.

---

## M8 — Corpus expansion, figures decision & hardening
**Goal:** depth and durability.

**Build:**
- Ingest the rest of the corpus in stages (Technical Summaries → WG chapters → Special Reports), **re-running eval at each step** to catch regressions.
- Decide **text-only** (index figure/table captions as text) vs **multimodal/structured** (figures & tables as first-class content) — now that you've felt how much caption text alone carries.
- Hardening: a test suite and a README that reports your eval results (this is the portfolio piece). (Observability and caching were set up in M7B.)

**Tools:** all of the above.

**Done when:** the full corpus is searchable without eval regressions, the figures decision is made and recorded, and someone else could run it from the README.

**Stop & teach:** how retrieval quality scaled with corpus size, and the reasoning behind the figures call.

---

### Reminder
No milestone begins without the human's go-ahead. The value of this project is the human understanding each step — the working tool is the byproduct.
