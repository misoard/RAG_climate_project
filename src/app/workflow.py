"""The walking skeleton: question in, cited Answer out, through the real architecture.

One ``run_sequential``: retrieve, then generate. Deliberately the thinnest pipeline
that touches every layer for real -- a genuine PDF, genuine local embeddings, the
real Gateway with the real re-prompt loop. Nothing here is mocked, which is the
whole point of building the skeleton before deepening any single component.

Retrieval quality is poor on purpose at this stage. The chunker cuts mid-sentence
and no metadata beyond page numbers is populated; M3 through M5 fix that, each
justified by the eval numbers M2 will start producing.

Run it (needs OPENROUTER_API_KEY in .env), from the repo root:
    uv run python -m app.workflow "How much has global surface temperature risen?"
"""

from __future__ import annotations

import asyncio
import sys

from agentic_core import Gateway, State, run_sequential

from contracts.models import Answer, Retriever
from app.agents import generation_agent
from app.config import build_gateway, generation_alias
from app.steps import make_generate_step, make_retrieve_step

DEFAULT_QUESTION = "How much has global surface temperature risen since pre-industrial times?"


async def answer_question(
    question: str,
    *,
    retriever: Retriever | None = None,
    gateway: Gateway | None = None,
) -> Answer:
    """Answer one question from the corpus.

    ``retriever`` and ``gateway`` are injectable so tests can supply a tiny store and
    a FakeRouter-backed gateway and run the whole pipeline offline. In normal use both
    default to the real thing.
    """
    if retriever is None:
        # Imported lazily: this pulls in torch and a 2.2GB model, which an offline
        # test passing its own retriever must not pay for.
        from rag.store.corpus import build_store

        retriever = build_store()
    gateway = gateway or build_gateway()

    agent = generation_agent(gateway, model=generation_alias())
    state = State({"question": question})
    await run_sequential(state, [make_retrieve_step(retriever), make_generate_step(agent)])
    return state["answer"]


async def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    state_answer = await answer_question(question)

    print(f"\nQ: {question}\n")
    print(state_answer.text)
    print(f"\nrefused={state_answer.refused}  qualifiers_preserved={state_answer.qualifiers_preserved}")
    print(f"citations ({len(state_answer.citations)}):")
    for c in state_answer.citations:
        section = f" {c.section}" if c.section else ""
        print(f"  - {c.report} {c.document_layer}{section} p.{c.page}   [{c.chunk_id}]")


if __name__ == "__main__":
    asyncio.run(main())
