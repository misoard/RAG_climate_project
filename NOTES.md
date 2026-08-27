# NOTES.md — confirmed `agentic-core` API surface

Recorded in **M0** by reading the installed package, not the README.
Source read: `.venv/lib/python3.12/site-packages/agentic_core/` (version `0.1.0`,
git `misoard/agentic-core`, `requires-python >=3.11`) plus `how_to_start/` from the
repo. Every signature below was additionally **exercised offline** against a
`FakeRouter` before being written down (see "Verification" at the end).

Re-check this file whenever the pinned commit of `agentic-core` moves.

---

## 0. Import discipline

`agentic_core/__init__.py` says it plainly: import from the **top level**
(`from agentic_core import Gateway, Agent`) — deep submodule paths are internal and
may move. The one exception is the eval harness, deliberately **not** re-exported
because it would shadow the builtin `eval`:

```python
from agentic_core import Agent, Gateway, State, run_sequential, run_concurrent, run_conditional
from agentic_core import Deployment, Settings, get_settings, configure
from agentic_core.eval import harness          # NOT `from agentic_core import eval`
from agentic_core.testing import FakeRouter, make_response
```

---

## 1. `Gateway` — the single model call path

```python
Gateway(
    *,
    router: RouterLike | None = None,       # inject a FakeRouter in tests
    settings: Settings | None = None,
    max_reprompts: int = 2,
    registry: dict[str, Deployment] | None = None,
    fallbacks: dict[str, list[str]] | None = None,
) -> Gateway

async Gateway.complete(
    messages: list[dict],                    # OpenAI-format; `Messages` alias
    *,
    model: str | None = None,                # a registry ALIAS ("fast"/"smart"), not a slug
    response_model: type[BaseModel] | None = None,
    **opts,
) -> Completion
```

Everything is keyword-only except `messages`. Notes that matter for us:

- **`model` is an alias**, resolved by the Router via the registry. Passing a raw
  provider slug is not the intended path.
- **`response_model` is what buys us the re-prompt loop.** The Gateway appends a
  system message containing the model's JSON Schema, validates the reply with
  `model_validate_json`, and on `ValidationError` re-prompts with the bad output
  *and the validation error* attached. Bounded by `max_reprompts` (default 2 → up
  to 3 attempts). Exhausted → `MalformedOutputError`.
- The Gateway owns **only** the malformed-output loop. Transient network retries
  and fallbacks belong to the Router underneath; when the Router raises, the
  Gateway classifies and surfaces (`PermanentError` / `AllModelsFailedError`) and
  never retries.
- It strips ``` fences before parsing, so a fenced JSON reply still validates.

**`Completion`** (a frozen dataclass, *not* a Pydantic model — `agentic_core.core.schemas`):
`.text`, `.model`, `.parsed` (the validated output, or `None`), `.usage`
(`TokenUsage`), `.cost_usd`, `.latency_ms`, **`.reprompt_attempts`** (0 == valid
first try), `.tool_calls`, `.extra`.
`.reprompt_attempts` is how M1 proves the re-prompt loop actually fired.

## 2. `Agent` — a typed, validated single call

```python
Agent(
    *,
    name: str,
    gateway: Gateway,
    output_model: type[OutputT],
    system_prompt: str,
    user_template: str,
    input_model: type[InputT] | None = None,
    model: str | None = None,               # alias; None -> settings.default_model ("fast")
)

async Agent.run(inp: InputT | dict, *, history: Messages | None = None, **opts) -> Completion[OutputT]
```

- **`Agent.run` returns a `Completion`, not the output model.** The typed answer is
  `completion.parsed`. (Easy to get wrong — `how_to_start/workflow.py` does
  `result.parsed`.)
- `system_prompt` / `user_template` are **strings passed at construction**, not file
  paths. `user_template` is rendered with `str.format(**inp.model_dump())`, so every
  `{placeholder}` must be a field of the input model — and any *literal* brace in a
  prompt has to be escaped. Prompt-loading from YAML is the consumer's job:
  `how_to_start/agents.py` has a ~10-line `load_prompt()` + `PromptSpec` we should
  copy into `app/` rather than expect from the package.
- Agents are **stateless**; conversation history is passed in per call.
- Override `render_messages(inp, *, history=None)` when formatting needs more than
  `.format()` — likely for us, since the context block is assembled from chunks.

## 3. Orchestration — plain async steps over any state

```python
State(data: dict)                                   # __getitem__/__setitem__/.get
Step = Callable[[StateT], Awaitable[ResultT]]       # async (state) -> result

async run_sequential(state, steps) -> list[Any]
async run_concurrent(state, steps, *, return_exceptions: bool = False) -> list[Any]
async run_conditional(state, predicate, if_true, if_false=None) -> ResultT | None
```

- The runner imports **no** `Agent` and knows nothing about the gateway — it just
  threads state through async callables. This is exactly why our retrieval call is a
  plain step and not an Agent: it makes no model call, so it has no business being one.
- `State` is a convenience bag, not a requirement — the combinators are generic over
  the state type, so a Pydantic state model would work too.
- **Both branches of `run_conditional` must be `async`** (it awaits the chosen one).
  A plain lambda raises `TypeError: object ... can't be used in 'await' expression`.
- `run_concurrent`: treat state as read-only inside concurrent steps and merge the
  *returned* results — concurrent writes to the shared bag race.

## 4. Eval harness — `agentic_core.eval.harness`

```python
@dataclass EvalCase(id: str, input: Any, expected: Any = None, metadata: dict = {})
@dataclass Score(name: str, value: float, passed: bool, detail: str | None = None)   # value normalized 0..1
@dataclass CaseResult(case_id, output=None, scores=[], error=None)
@dataclass EvalReport(results: list[CaseResult]); .summary() -> {"n_cases", "errors", "scorers": {name: {"mean","pass_rate","n"}}}

Task   = Callable[[Any], Awaitable[Any]]            # async (case.input) -> output
Scorer = Callable[[Any, EvalCase], Score | Awaitable[Score]]   # (output, case) -> Score; sync OR async

async run_eval(dataset, task, scorers, *, concurrency: int = 8) -> EvalReport
compare(baseline: EvalReport, candidate: EvalReport) -> dict   # per-scorer mean deltas (candidate - baseline)
```

Built-in scorers: `exact_match`, `numeric_close(tolerance=...)` (a factory), and
`llm_judge(gateway, *, rubric: str, model: str | None = None)` (a factory returning
an async scorer, with a `JudgeVerdict` model of `passed`/`score`/`reasoning`).

For M2 this means: our four metrics (retrieval hit rate, faithfulness, refusal
correctness, uncertainty preservation) are each **a `Scorer` we write**, returning a
`Score` normalized to 0..1. The harness supplies the runner, the aggregation, and
`compare()` — the regression check. A case that raises is captured in
`CaseResult.error` and **excluded from the means**, so a broken case can't silently
flatter a metric; watch the `errors` count in every report.

## 5. `FakeRouter` — the offline test seam

```python
from agentic_core.testing import FakeRouter, make_response

FakeRouter(behaviours: list)     # each item: a response object (returned) or an Exception (raised)
FakeRouter.calls                 # list[{"model", "messages", "kwargs"}] — every call recorded
make_response(content: str, *, model=..., prompt_tokens=5, completion_tokens=3, cost=0.00012)
```

Injected as `Gateway(router=FakeRouter([...]))`. Because behaviours are a scripted
list, "malformed, then valid" is a two-item list — which is precisely how we test the
re-prompt loop offline, with no key and no network. Running out of behaviours raises
`AssertionError`, so an unexpected extra model call fails loudly.

## 6. Config — models as data

`Settings` (pydantic-settings, reads `.env`): `openrouter_api_key`, `openai_api_key`,
`anthropic_api_key`, `default_model="fast"`, `request_timeout_s=30.0`, `num_retries=2`,
`llm_live_tests=False`, `logfire_token`, `otel_console_export`, `jaeger_endpoint`.
`Deployment(alias, model, params={}, rpm=None, tpm=None)`.
Provider keys resolve by convention: model prefix `X/...` → `Settings.X_api_key`.

`DEFAULT_REGISTRY` / `DEFAULT_FALLBACKS` in the package are explicitly labelled
*illustrative scaffolding* — a consumer defines its own and passes `registry=`.
`how_to_start/config.py` is the pattern to mirror in `app/config.py`: subclass
`Settings` as `AppSettings` with env-driven model slugs, a `build_registry(settings)`,
a `FALLBACKS` dict, and `build_gateway(settings=None) -> Gateway`.

## 7. Observability & guardrails

```python
configure(settings: Settings | None = None, *, service_name: str = "agentic-core") -> bool   # True if wired
```
Opt-in, called **once at startup**. Priority: Logfire token → Jaeger/OTLP endpoint →
console. With none set it is a no-op and spans cost nothing, so tests stay offline.
The Gateway already emits one `llm.completion` span per call; our own retrieval spans
are ours to add.

Guardrails (`from agentic_core import ...`), all returning `GuardResult(ok, guard, reason, value)`
with `.raise_if_failed()`:
`check_injection(text, *, patterns=None)`, `check_injection_llm(text, gateway, *, model=None)` (async),
`validate_tool_args(model, raw_args)`, `scan_pii(text, *, redact=True)`,
`enforce_policy(text, *, blocklist: Iterable[str])`.
`enforce_policy` is a **case-insensitive substring blocklist** — that is the whole
mechanism. Useful for the "assessed ≠ recommended" output policy (M6), but it can only
catch literal phrases; the real work there will be the judge agent, not this.

---

## Deltas from `CLAUDE.md` — where the code differs from the sketch

Nothing contradicts the architecture, but four details in `CLAUDE.md §3/§6` are
imprecise. Following the code:

1. **`Agent.run()` returns `Completion[OutputT]`, not `Answer`.** Our workflow must
   read `.parsed` (and `.reprompt_attempts` is what proves the Gateway re-prompted).
2. **`§3` points at `guardrails/io_guards.py` and `eval/harness.py` as paths.** Both
   exist, but the supported import is top-level for guardrails
   (`from agentic_core import check_injection, enforce_policy`) and
   `from agentic_core.eval import harness` for the harness — never
   `from agentic_core import eval`, which would shadow the builtin.
3. **`run_conditional` needs two async branches.** The "retrieval too weak → refuse"
   branch in `§3` must be written as an `async def`, not a lambda.
4. **Prompts-as-files is not in the package.** `§5` gives us `app/prompts/`, and the
   `PromptSpec` + `load_prompt()` loader to fill it has to be copied from
   `how_to_start/agents.py` into our own `app/agents.py`.

5. **The Gateway keeps its router private.** It is passed as `Gateway(router=...)`
   but stored as `_router`, so a test that wants to inspect calls must keep its own
   reference to the `FakeRouter` rather than reading it back off the gateway.
6. **The schema instruction is appended as the *last* message.** `router.calls[i]
   ["messages"][-1]` is the Gateway's "respond with JSON matching this schema" system
   message, *not* the rendered user prompt. To assert on what the agent actually said,
   join all messages or select by `role`. (Both of these were found by M1's tests
   failing, not by reading -- worth the reminder that this file is notes, not proof.)

One contract observation from our side: `Answer.citations` and
`Answer.supporting_chunk_ids` default to empty lists, which is what makes a refusal
(`refused=True`, no citations) valid output rather than a schema violation. That is
deliberate — see `src/contracts/models.py`.

---

## Verification

An offline script exercised all of the above with `FakeRouter` (no key, no network),
using **our own M0 contracts** as the Agent's `input_model`/`output_model`:

| Check | Result |
|---|---|
| `Agent(GenerationInput -> Answer)` through `Gateway` | `Completion.parsed` is an `Answer` |
| Malformed reply, then valid | `reprompt_attempts == 1`, 2 router calls — the re-prompt loop fires |
| A dummy retriever vs. our `Retriever` Protocol | `isinstance(...)` True (runtime-checkable, structural) |
| `run_sequential(retrieve_step, generate_step)` over `State` | 2 results, `State["answer"]` is an `Answer` |
| `run_conditional` refusal branch | returns `Answer(refused=True)` |
| `harness.run_eval` + `harness.compare` | scored report and per-scorer deltas |
