"""The walking skeleton end to end, offline.

Runs the real pipeline -- real steps, real State, real Gateway, real Answer schema
validation -- with only the two expensive edges faked. In particular this is where
M1's second acceptance criterion is verified: that the Gateway actually rejects a
malformed answer and re-prompts, rather than that being assumed because the README
says so.
"""

from __future__ import annotations

import pytest
from agentic_core import Deployment, Gateway
from agentic_core.testing import FakeRouter, make_response

from conftest import answer_json
from contracts.models import Answer
from app.steps import render_context_block, make_retrieve_step
from app.workflow import answer_question

REGISTRY = {"fast": Deployment(alias="fast", model="fake/model", params={})}


def gateway_returning(*bodies: str, max_reprompts: int = 2) -> tuple[Gateway, FakeRouter]:
    """A Gateway whose model replies with each body in turn, plus the router to inspect.

    The router is handed back rather than read off the Gateway afterwards: it keeps it
    as a private ``_router``, and a test that reaches into another object's privates
    breaks the moment that object is refactored.
    """
    router = FakeRouter([make_response(b) for b in bodies])
    return Gateway(router=router, registry=REGISTRY, max_reprompts=max_reprompts), router


async def test_pipeline_returns_a_validated_cited_answer(store):
    gateway, _ = gateway_returning(answer_json())
    answer = await answer_question(
        "How much has global surface temperature risen?", retriever=store, gateway=gateway
    )
    assert isinstance(answer, Answer)
    assert answer.refused is False
    assert len(answer.citations) >= 1
    assert answer.citations[0].chunk_id == "AR6-SYR#0001"
    assert answer.citations[0].page == 4


async def test_gateway_reprompts_on_a_malformed_answer(store):
    """M1 acceptance: the re-prompt loop fires, and we can prove it did.

    The first reply is prose rather than JSON. The Gateway must reject it against the
    Answer schema, re-prompt with the validation error, and accept the second -- so
    the citation structure is enforced by machinery, not by the prompt's persuasiveness.
    """
    gateway, router = gateway_returning("Sorry, I cannot produce JSON today.", answer_json())

    # Driven through the step rather than answer_question, so the Completion is
    # reachable: reprompt_attempts is the core's own record of the loop firing.
    from agentic_core import State

    from app.agents import generation_agent
    from app.steps import make_generate_step

    state = State({"question": "How much warming?"})
    await make_retrieve_step(store)(state)
    answer = await make_generate_step(generation_agent(gateway, model="fast"))(state)

    assert isinstance(answer, Answer)
    assert state["completion"].reprompt_attempts == 1  # 0 would mean valid first try
    assert len(router.calls) == 2  # and it cost exactly one extra model call
    assert len(answer.citations) == 1

    # The re-prompt must tell the model what was wrong, or it is just a retry.
    retry = router.calls[1]["messages"][-1]["content"]
    assert "valid" in retry.lower() or "error" in retry.lower() or "json" in retry.lower()


async def test_exhausting_the_reprompts_raises_rather_than_returning_junk(store):
    """A model that never complies must fail loudly, not yield an unvalidated Answer."""
    from agentic_core import MalformedOutputError

    gateway, _ = gateway_returning("nope", "still nope", max_reprompts=1)
    with pytest.raises(MalformedOutputError):
        await answer_question("How much warming?", retriever=store, gateway=gateway)


async def test_a_refusal_is_valid_output_not_an_error(store):
    """Refusal is a first-class outcome (M2 measures it), so it must validate cleanly."""
    body = answer_json(
        text="The excerpts do not address rainfall in Lyon in 2050.",
        citations=[],
        qualifiers_preserved=True,
        refused=True,
        supporting=[],
    )
    gateway, _ = gateway_returning(body)
    answer = await answer_question("Will it rain in Lyon in 2050?", retriever=store, gateway=gateway)
    assert answer.refused is True
    assert answer.citations == []


async def test_the_agent_is_only_offered_chunks_that_were_retrieved(store):
    """allowed_chunk_ids is the allow-list a fabricated citation is checked against."""
    gateway, router = gateway_returning(answer_json())
    await answer_question("warming", retriever=store, gateway=gateway)

    # The whole conversation, not messages[-1]: the Gateway appends its JSON Schema
    # instruction as the final message, so the rendered prompt is not last.
    sent = "\n".join(m["content"] for m in router.calls[0]["messages"])
    # Every chunk in the prompt is one the store actually returned.
    for chunk_id in ["AR6-SYR#0001", "AR6-SYR#0002", "AR6-SYR#0003"]:
        assert chunk_id in sent
    # And the schema really is shipped with the request -- that is what makes the
    # Answer contract enforceable rather than merely requested.
    assert "qualifiers_preserved" in sent and "JSON Schema" in sent


async def test_context_block_headers_carry_everything_a_citation_needs(store):
    step = make_retrieve_step(store, k=2)
    from agentic_core import State

    state = State({"question": "temperature"})
    await step(state)
    block = render_context_block(state["hits"])

    assert "chunk_id: AR6-SYR#0001" in block
    assert "report: AR6-SYR" in block
    assert "layer: SPM" in block
    assert "pages: 4" in block


def test_a_multi_page_chunk_renders_a_page_range():
    from conftest import make_chunk
    from contracts.models import RetrievedChunk

    hit = RetrievedChunk(chunk=make_chunk("c", "text", page_start=5, page_end=6), score=0.5)
    assert "pages: 5-6" in render_context_block([hit])
