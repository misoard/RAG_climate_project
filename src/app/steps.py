"""The orchestration steps: retrieval, then generation, over a shared ``State``.

The architectural point of this file (CLAUDE.md 3) is what retrieval is *not*.
Retrieval makes no model call -- it is a matrix multiply against local vectors --
so it is a plain async step, not an ``Agent``. Making it an Agent would buy nothing
and would imply a cost and a failure mode it does not have. The core's combinators
know nothing about agents either; they just thread state through async callables,
which is exactly why the two halves compose without either knowing about the other.

Both steps are built by factories so their dependencies (the retriever, the agent)
are injected rather than reached for globally -- that is what lets the workflow test
run entirely offline against a FakeRouter.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agentic_core import Agent, State

from contracts.models import Answer, GenerationInput, RetrievedChunk, Retriever
from rag.settings import load_settings

Step = Callable[[State], Awaitable[object]]


def render_context_block(hits: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into the text the agent sees, one header per excerpt.

    The header is the mechanism behind every citation: the model cannot cite what it
    cannot see, so chunk_id, report, layer, section and page range are printed beside
    each excerpt in a fixed format for it to copy verbatim. Anything omitted here is
    something the model would have to invent.

    The page *range* is shown rather than a single page because a chunk may straddle a
    break, and the model needs to know which pages are legitimately citable for it.
    """
    blocks = []
    for hit in hits:
        c = hit.chunk
        pages = f"{c.page_start}" if c.page_start == c.page_end else f"{c.page_start}-{c.page_end}"
        blocks.append(
            f"[chunk_id: {c.chunk_id} | report: {c.report} | layer: {c.document_layer} "
            f"| section: {c.section or 'unknown'} | pages: {pages}]\n{c.text.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def make_retrieve_step(retriever: Retriever, *, k: int | None = None) -> Step:
    """Build the step that turns ``state["question"]`` into ranked chunks."""
    top_k = k if k is not None else load_settings().retrieval.top_k

    async def retrieve_step(state: State) -> list[RetrievedChunk]:
        hits = await retriever.retrieve(state["question"], k=top_k)
        state["hits"] = hits
        return hits

    return retrieve_step


def make_generate_step(agent: Agent[GenerationInput, Answer]) -> Step:
    """Build the step that turns retrieved chunks into a validated ``Answer``."""

    async def generate_step(state: State) -> Answer:
        hits: list[RetrievedChunk] = state.get("hits", [])
        generation_input = GenerationInput(
            question=state["question"],
            context_block=render_context_block(hits),
            # The allow-list travels with the request so a fabricated citation is
            # caught by set membership after the fact (M6), not by trusting the prompt.
            allowed_chunk_ids=[h.chunk.chunk_id for h in hits],
        )
        completion = await agent.run(generation_input)
        # Agent.run returns a Completion, not the output model -- the typed value is
        # .parsed. Easy to get wrong; NOTES.md flags it.
        state["completion"] = completion
        state["answer"] = completion.parsed
        return completion.parsed

    return generate_step
