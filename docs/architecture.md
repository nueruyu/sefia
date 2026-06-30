# Architecture map

A navigation aid for humans and AI coding agents: where things live, which package
may import which, and **"if you want to change X, look at Y."** For *how* the runtime
works, see [how-it-works.md](./how-it-works.md); this doc is the layout.

## Repository shape

A `uv` workspace (`pyproject.toml` → `[tool.uv.workspace]`) of small packages:

| Package | Path | Responsibility |
| --- | --- | --- |
| **sefia** | `packages/sefia` | The core: `@infer`, the inference loop, tool model, sessions, the default LLM strategy. |
| **sefios** | `packages/sefios` | Official batteries: `SessionScope`, default policies/middleware/handlers, ready-made tools. |
| **sefia_litellm** | `packages/sefia_litellm` | Provider adapter — an `LLMClient` implemented over LiteLLM. |
| **jsonstream** | `packages/jsonstream` | Standalone incremental JSON parser (used for streaming tool args). Zero deps. |
| **examples** | `examples` | Runnable end-to-end agents. |
| **glyff** | *(separate repo)* | Content-addressed durable execution. A dependency, not vendored. |

## Dependency direction (one-way, no cycles)

```
pydantic ─┐
glyff ────┤
jsonstream┴─▶ sefia ─▶ sefia_litellm
                │            ▲
                └─▶ sefios ──┘  (optional: sefios[litellm])
                       ├─▶ ddgs        (optional: sefios[web])
                       └─▶ typer/rich  (optional: sefios[cli])
```

Rules that keep the layering clean — worth preserving in any change:

- **sefia core never imports sefios, sefia_litellm, or any provider SDK.** Provider
  and convenience concerns live *above* the core.
- **Provider specifics stay in adapters.** Anything LiteLLM/OpenAI-shaped belongs in
  `sefia_litellm`, behind the `LLMClient` interface. The core only knows the interface.
- **`sefios` depends on `sefia` only**; its provider/web/cli needs are **optional
  extras**, so installing `sefios` alone pulls in nothing heavy.

## Inside `sefia` (the core)

Modules with a leading underscore are internal; the public surface is whatever
`packages/sefia/src/sefia/__init__.py` re-exports.

| Module | Responsibility | Key symbols |
| --- | --- | --- |
| `_decorators.py` | The entry points. Calling `@infer` builds the executor and engraves the run. | `infer`, `tool`, `policy`, `profile` |
| `_executor.py` | The step loop, tool execution, middleware composition. | `InferenceExecutor` |
| `inference.py` | Plain data: the decision/history types and the call descriptor. | `FunctionInfo`, `ToolCallDecision`, `FinalAnswerDecision` |
| `_session.py` | Wraps a `glyff.Session`, builds the strategy, installs the context. | `Session` |
| `_context.py` | The contextvar-scoped run state; call-scoped state stores. | `SessionContext`, `get_context`, `get_call_state_store` |
| `_profiles.py` / `_metadata.py` | Per-call model/policy selection; the `__sefia_metadata__` store. | `Profile` |
| `_tool_system.py` | The tool registry and the collector interface. | `Tool`, `ToolRegistry`, `ToolCollector` |
| `tool_collectors/_collector.py` | Default discovery from the held objects. | `DefaultToolCollector` |
| `_toolify.py` | Wrap non-decorated callables/objects as tools. | `toolify`, `Toolset` |
| `_state_store.py` / `stores/` | Typed state persistence; memory & file backends. | `StateStore`, `MemorySessionStore`, `FileSessionStore` |
| `event_system.py` / `events.py` | Observation seam: publisher + event types. | `EventPublisher` |
| `_markers.py` / `streaming.py` | `AsRawText`; the tool-arg streaming side channel. | `AsRawText`, `ArgStream`, `StringDelta` |
| `llm/` | The **default** `InferenceStrategy`: function → prompt+schema → decision. | `LLMInferenceStrategy`, `LLMClient`, prompt formatters |
| `pydantic/` | The **default** `ModelInspector`: schema gen & validation via Pydantic. | `PydanticModelInspector` |

### The seams (`_interfaces/`) — the extension ports

These ABCs are where you swap behavior without touching the core. Each has a default
implementation noted in parentheses.

| Interface | Swap to… | Default |
| --- | --- | --- |
| `InferenceStrategy` | replace the "brain" (a different prompting scheme, or non-LLM) | `llm/LLMInferenceStrategy` |
| `LLMClient` (in `llm/_client.py`) | add an LLM provider | `sefia_litellm.LiteLLMClient` |
| `ModelInspector` | non-Pydantic schema/validation | `pydantic/PydanticModelInspector` |
| `SessionStore` | a new persistence backend | `stores/Memory`/`File` |
| `ToolCollector` | a different tool-discovery rule | `DefaultToolCollector` |
| `Policy` + `InferenceMiddleware`/`StepMiddleware` | control: retries, caps, guards | `sefios` middleware/policies |

## Inside `sefios` (the batteries)

| Path | Responsibility |
| --- | --- |
| `_scope.py` | `SessionScope` — the configured front door that wires client + glyff + store + defaults. |
| `policies/` | `DefaultPolicy` (step cap, default handlers) and a `CustomPolicy` builder. |
| `middleware/` | `_max_steps`, `_retry`, `_stagnation` — control-seam behaviors. |
| `handlers/` | `_cost` — an observation-seam handler (cost accounting). |
| `tools/` | `human.py` (HITL pause-by-raise), `web.py` (DuckDuckGo search). |
| `state.py` | App-level state helpers: `StateRegistry`, `StateContainer`, `state`, `get_state`. |

## "If you want to change X, look at Y"

| Goal | Where |
| --- | --- |
| Add an LLM provider | implement `LLMClient`; mirror `packages/sefia_litellm/src/sefia_litellm/_client.py` |
| Change how the prompt / decision schema is built | `llm/_strategy.py` (the `_ExecutionDirector`s), `llm/_xml_prompt_formatter.py` |
| Add a built-in tool | `packages/sefios/src/sefios/tools/` |
| Add retry / step-cap / a guard | a `Policy` + `StepMiddleware`/`InferenceMiddleware` in `sefios/middleware/` |
| Observe runs (logging, tracing, cost) | a handler over `events.py`; see `sefios/handlers/_cost.py` |
| Add a persistence backend | implement `SessionStore`; reference `stores/_file.py` |
| Change which methods are tools | `tool_collectors/_collector.py` |
| Per-call model/policy switch | `Profile` + the `@profile` decorator |
| Support a new output type system | `ModelInspector` in `pydantic/_model_inspector.py` |
| Trace the runtime end to end | [how-it-works.md](./how-it-works.md) |

## Conventions

- **Underscore = internal.** `_module.py` and `_Symbol` are not API; import the public
  names from a package's `__init__.py`.
- **Interfaces live in `_interfaces/`** as ABCs; concrete defaults live in
  feature folders (`llm/`, `pydantic/`, `stores/`, `tool_collectors/`).
- **Tests mirror source** under each package's `tests/units/` (per-module) and
  `tests/scenarios/` (behavioral). Add tests next to the layer you change.
- **The two seams are deliberately separate:** *middleware* controls (can retry /
  short-circuit), *handlers* only observe (their exceptions are isolated). Don't route
  control through a handler.
- **Keep the dependency arrows one-way** (see above). New provider/IO concerns go in
  an adapter or in `sefios`, never in `sefia` core.

## See also

- [how-it-works.md](./how-it-works.md) — the runtime data flow these modules implement.
- [DESIGN.md](../DESIGN.md) — why the boundaries are where they are.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — setup, commands, and the development workflow.
