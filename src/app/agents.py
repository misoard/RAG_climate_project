"""The typed agents this project runs. The generation half's policy layer.

``agentic_core`` owns the ``Agent`` machinery; this file says which agents exist
and what their contracts are. The prompt loader below is copied from the core's
``how_to_start/agents.py`` on purpose -- prompts-as-files is a consumer concern
that the package deliberately does not ship (see NOTES.md).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from agentic_core import Agent, Gateway

from contracts.models import Answer, GenerationInput

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptSpec(BaseModel):
    """Validated contract for a prompt file -- fail fast on malformed YAML.

    A prompt file is untrusted config like any other, so it gets boundary validation:
    a missing key raises a ValidationError naming the field at load, rather than a
    KeyError much later during agent construction.
    """

    system: str
    user_template: str
    version: int = 1


def load_prompt(name: str) -> PromptSpec:
    """Load a versioned prompt from ``prompts/<name>.yaml``."""
    return PromptSpec.model_validate(yaml.safe_load((_PROMPTS_DIR / f"{name}.yaml").read_text()))


def generation_agent(gateway: Gateway, *, model: str | None = None) -> Agent[GenerationInput, Answer]:
    """The agent that turns retrieved context into a cited, qualified Answer.

    This is where the M0 contracts finally earn their keep: because ``Answer`` is the
    ``output_model``, the Gateway ships its JSON Schema with the request and validates
    the reply against it, re-prompting with the validation error when it doesn't fit.
    The citation structure and the qualifiers_preserved/refused flags are therefore
    enforced by the machinery rather than by hoping the prompt was persuasive.
    """
    prompt = load_prompt("generation")
    return Agent(
        name="generation",
        gateway=gateway,
        input_model=GenerationInput,
        output_model=Answer,
        model=model,
        system_prompt=prompt.system,
        user_template=prompt.user_template,
    )
