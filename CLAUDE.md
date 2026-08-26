# CLAUDE.md — IPCC Climate Q&A (RAG)

A grounded, cited question-answering tool over the IPCC AR6 corpus. This is a **learning project**: the goal is not just a working tool, but that the human understands every step. Read this whole file before doing anything, then follow `MILESTONES.md` in order.

---

## 0. How you must work here — the learning protocol (READ FIRST)

This overrides the usual instinct to finish the whole task in one go.

- **Work one milestone at a time, in the order in `MILESTONES.md`.** Never start the next milestone on your own.
- **At the end of every milestone: STOP and run a Milestone Review**, then wait for the human to explicitly say to continue. The review must contain:
  1. **What I built** — the files/functions added or changed, in a sentence each.
  2. **The concepts** — the 1–3 ideas this milestone was really about, explained plainly (this is the point of the project).
  3. **Decisions & alternatives** — what you chose, what you rejected, and why. Name the trade-off.
  4. **How to run & verify it** — exact commands, and what the human should look at.
  5. **Eval delta** — once the eval harness exists (M2+), the before/after numbers. "It feels better" is never acceptable.
- **Within a milestone, work in small, reviewable commits** with clear messages, and narrate what you're doing and why as you go.
- **Do not skip ahead, and do not optimize a stage before the M2 eval gate exists.** No reranking, hybrid search, or fancy chunking until it can be measured.
- **Favor teaching clarity over cleverness.** Prefer obvious code with a comment on the *why* over dense code that's faster to write.
- If you think a later milestone's work is needed early, **stop and ask** rather than pulling it forward.

---

## 1. What this is (scope)

Answer open-ended climate questions **only** from the AR6 corpus, with every answer citing report + section + page, preserving the IPCC's confidence/likelihood qualifiers, and **refusing when the corpus doesn't support an answer**.

Explicitly **out of scope**: local/regional "where I live" projections (a data problem, not RAG), policy scorecards, live data, and anything implying the IPCC *recommends* a policy rather than *assesses* options. The IPCC is policy-relevant but not policy-prescriptive — never phrase an answer as an IPCC recommendation.

Success test: a journalist, teacher, or student can ask a real question and get an answer they could safely quote, with the citation and the uncertainty intact.

**Language (design intent).** The corpus is **English** — the only complete, authoritative edition (full WG chapters and the FAQ text used for the eval gold set exist in English only; the IPCC/GIEC translates just the SPM/SYR layer into French and other UN languages). English stays the single source of truth and the single embedding space. **French is a planned later extension, handled at the edges, not in the store:** accept a French question, retrieve cross-lingually against the English corpus, and generate the answer in French while citing the English source. The only decision this forces *now* is choosing a multilingual embedding model (§4), because re-embedding the whole corpus later is the one expensive-to-reverse step. Actual French I/O stays deferred (see M6/M7A notes).

---

## 2. Architecture principles

1. **Walking skeleton first.** The thinnest slice that runs end-to-end *through the real architecture* (M1), then deepen each component behind stable interfaces. Never build a component in isolation against an imagined interface.
2. **Contracts first.** The typed interfaces that cross the retrieval↔generation boundary are defined in M0 and kept stable. Everything else is a swappable implementation behind them.
3. **Order work by risk, not by layer.** The real unknowns — does the plumbing fit, and can we measure quality — are attacked first and cheaply. Parsing/reranking/etc. are deferred optimizations.
4. **Evaluate before you optimize.** After M2, every change is justified by a measured delta.
5. **Own the retrieval half; lean on `agentic-core` for the generation half.** See §3.

---

## 3. The `agentic-core` boundary (non-negotiable)

RAG has two halves. The retrieval half is **your code**, in a `rag/` package. The generation/reliability/orchestration/eval half runs on **`agentic-core`** (https://github.com/misoard/agentic-core). Keep the boundary clean:

| Concern | Where it lives |
|---|---|
| Ingestion, chunking, embeddings, vector store, reranking, metadata | `rag/` — **your code**, outside the core |
| A retrieval call in the pipeline | A plain **async step** over the orchestration `State` — **not an Agent** (it makes no model call) |
| Generation: context → cited answer | An **`Agent`** with a Pydantic output schema (`Answer`) |
| Enforcing well-formed citations / schema | The **`Gateway`** validation + re-prompt loop — free, use it |
| "Retrieval too weak → refuse" branch | `run_conditional` |
| Multi-query / per-working-group parallel retrieval | `run_concurrent` (only if M-retrieval eval justifies it) |
| Faithfulness / uncertainty-preservation judging | An LLM-judge `Agent`, and/or the eval harness's judge |
| Evaluation (hit rate, faithfulness, refusal) | The core's **`eval/harness.py`** — feed it, don't rebuild it |
| Tracing | The core's OTel spans (`configure()`); add retrieval spans yourself |
| Input injection check / output policy ("assessed ≠ recommended") | The core's `guardrails/io_guards.py` |

**Standing rule — confirm the real API.** These instructions were written from the `agentic-core` README, not its source. Before using any core primitive (`Agent`, `Gateway`, `run_sequential`, the eval harness, `FakeRouter`), **read the installed package and `how_to_start/` to confirm exact signatures and import paths.** Never assume a signature; verify it, and if it differs from what's described here, follow the code and tell the human.

Mirror the core's recommended app split (from `how_to_start/`): `config.py` (your model registry + `build_gateway`), `agents.py` (typed agents), `prompts/` (versioned prompts), `workflow.py` (entry point).

---

## 4. Tech stack / tools

- **Env & packaging:** `uv`. Add the core as a git dependency: `uv add "agentic-core @ git+https://github.com/misoard/agentic-core.git"`.
- **LLM calls:** always through `agentic-core`'s `Gateway`/`Agent`. Models are config (a `Deployment` registry), never hardcoded.
- **PDF parsing (M3):** `PyMuPDF` (fitz) for text+layout; consider `docling`/`unstructured` for the two-column layout, captions, and boxes. Manual QA is mandatory.
- **Embeddings:** a local **multilingual** `sentence-transformers` model (a multilingual BGE / E5 variant) to keep it free and private **and** to leave the French-facing door open. Cross-lingual retrieval (French query → English corpus) works well with these; an English-only model would force a full re-embed later. Fix this choice in M1.
- **Vector store:** M1 uses a trivial in-memory brute-force cosine store (`numpy`) — no external DB. Upgrade to `pgvector` or `Qdrant` only in the retrieval-quality milestone, behind the `Retriever` interface.
- **Reranker (retrieval milestone):** a `sentence-transformers` cross-encoder.
- **Tests:** `pytest`, fully offline. Test agents/workflows with the core's `FakeRouter` + `make_response` — no API key in tests.
- **Observability:** the core's `configure()` at startup; no-op if unconfigured.
- **Secrets:** `.env` only (e.g. `OPENROUTER_API_KEY`). Never commit or hardcode keys.

---

## 5. Target project structure

Grown milestone by milestone — do **not** create it all at once.

```
ipcc-rag/
├── CLAUDE.md
├── MILESTONES.md
├── pyproject.toml            # uv; depends on agentic-core
├── .env.example
├── config/
│   └── settings.yaml         # chunk sizes, top-k, thresholds, model aliases
├── data/
│   ├── raw/                  # IPCC PDFs (git-ignored)
│   ├── processed/            # parsed chunks (git-ignored)
│   ├── manifest.json
│   └── eval/gold_set.jsonl
├── src/
│   ├── contracts/            # M0 — the stable boundary types (see §6)
│   │   └── models.py
│   ├── rag/                  # YOUR retrieval half
│   │   ├── ingestion/        # download.py, parse.py, metadata.py
│   │   ├── chunking/chunker.py
│   │   ├── embeddings/embedder.py
│   │   ├── store/            # inmemory.py (M1) → vector_store.py (later)
│   │   ├── retrieval/        # retriever.py, router.py, normalize.py, rerank.py
│   │   └── faithfulness/checks.py
│   └── app/                  # mirrors agentic-core how_to_start split
│       ├── config.py         # model registry + build_gateway()
│       ├── agents.py         # generation agent, judge agent
│       ├── prompts/          # versioned prompt templates
│       ├── steps.py          # retrieval as an async orchestration step
│       └── workflow.py       # the run_sequential pipeline (entry point)
├── eval/
│   ├── build_gold_set.py     # harvest AR6 chapter FAQs
│   └── run_eval.py           # drives agentic_core eval harness
├── interface/                # thin UI (later milestone)
└── tests/
```

---

## 6. The contracts (define in M0, keep stable)

Concrete starting definitions — refine names in M0 but keep the shape. These are the interfaces the two halves meet at.

```python
from typing import Literal, Protocol
from pydantic import BaseModel

# ---- the unit of retrieval ----
class Chunk(BaseModel):
    chunk_id: str
    text: str
    report: str                                    # "AR6-SYR", "AR6-WG1", ...
    working_group: Literal["WG1","WG2","WG3","SYR","SR1.5","SROCC","SRCCL"]
    document_layer: Literal["SPM","TS","chapter","FAQ"]
    chapter: str | None = None
    section: str | None = None
    page: int
    is_headline_statement: bool = False
    confidence_terms: list[str] = []               # e.g. ["high confidence"]
    likelihood_terms: list[str] = []               # e.g. ["very likely"]
    scenarios_mentioned: list[str] = []            # e.g. ["SSP1-2.6"]
    figure_or_table_ref: str | None = None

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float

# ---- the retrieval boundary ----
class Retriever(Protocol):
    async def retrieve(
        self, query: str, k: int = 8, filters: dict | None = None
    ) -> list[RetrievedChunk]: ...

# ---- the generation boundary (the Agent's typed I/O) ----
class Citation(BaseModel):
    report: str
    document_layer: str
    section: str | None = None
    page: int

class GenerationInput(BaseModel):
    question: str
    context_block: str            # rendered from retrieved chunks (with inline source markers)
    allowed_chunk_ids: list[str]  # for post-hoc citation validation

class Answer(BaseModel):
    text: str
    citations: list[Citation]
    qualifiers_preserved: bool    # did the answer keep IPCC confidence/likelihood language?
    refused: bool                 # true when the corpus doesn't support an answer
    supporting_chunk_ids: list[str]
```

The orchestration `State` carries: `question → list[RetrievedChunk] → Answer`. The generation `Agent` uses `GenerationInput` as `input_model` and `Answer` as `output_model`, so the Gateway's re-prompt loop enforces the citation/qualifier structure for free.

---

## 7. Coding conventions

- Typed throughout; Pydantic models at every boundary; `async` call paths (the core is async).
- Small single-responsibility functions with a comment on the *why* where it isn't obvious.
- No secrets in code. Models via the `Deployment` registry, keys via `.env`.
- Every non-trivial component gets a test; agent/workflow tests run offline via `FakeRouter`.
- One logical change per commit, message explaining the *why*.
```
