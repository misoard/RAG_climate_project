"""This project's model configuration -- policy, not machinery.

``agentic_core`` owns the *shape* of config (``Settings``, ``Deployment``, the
``Gateway``); this file owns the *values*. It is the counterpart to the core's
``core/config.py``, and mirrors the split in the core's ``how_to_start/config.py``.

The rule it exists to enforce (CLAUDE.md 4 and 7): no model name is ever hardcoded
at a call site. Agents ask for an alias -- "generation", resolved via the registry
-- so swapping the underlying model is an env change, and no code knows a provider
slug except this file.
"""

from __future__ import annotations

from agentic_core import Deployment, Gateway, Settings

from rag.settings import load_settings


class AppSettings(Settings):
    """Base Settings (keys, timeouts, .env loading) plus this project's model slugs.

    Slugs are env-driven so a model swap never touches code; params like temperature
    stay in ``build_registry`` below, because they are structural to the task rather
    than varying by environment.
    """

    fast_model: str = "openrouter/openai/gpt-4o-mini"
    smart_model: str = "openrouter/anthropic/claude-sonnet-4.6"


def build_registry(settings: AppSettings) -> dict[str, Deployment]:
    """The alias table: aliases from settings.yaml, slugs from env, params from code.

    ``temperature=0.0`` throughout, and not as a stylistic default. This tool answers
    from a fixed corpus and must be quotable; sampling variation would mean the same
    question yields differently-worded claims from the same evidence, which makes both
    the M2 eval and any citation check unreproducible.
    """
    return {
        "smart": Deployment(alias="smart", model=settings.smart_model, params={"temperature": 0.0}),
        "fast": Deployment(alias="fast", model=settings.fast_model, params={"temperature": 0.0}),
    }


# Resilience graph: if the generation model is unavailable, degrade rather than fail.
FALLBACKS: dict[str, list[str]] = {"smart": ["fast"]}


def generation_alias() -> str:
    """Which registry alias the generation agent runs on (config/settings.yaml)."""
    return load_settings().models.generation


def build_gateway(settings: AppSettings | None = None) -> Gateway:
    """Construct the one model call path this project uses."""
    settings = settings or AppSettings()
    return Gateway(
        settings=settings,
        registry=build_registry(settings),
        fallbacks=FALLBACKS,
    )
