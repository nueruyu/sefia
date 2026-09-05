# Architecture map

A navigation aid for contributors: where things live, which package may import which,
and **where to change what**. For *how* the runtime works, see
[how-it-works.md](./how-it-works.md); this doc is the layout.

## Repository shape

A `uv` workspace (`pyproject.toml` → `[tool.uv.workspace]`) of small packages:

| Package | Path | Responsibility |
| --- | --- | --- |
| **sefia** | `packages/sefia` | The core: `@infer`, the inference loop, tool model, sessions, the default LLM strategy. |
| **sefios** | `packages/sefios` | Opinionated batteries and integration layer: `SessionScope`, default policies/middleware/handlers, ready-made tools, and the extra-gated CLI/HTTP facades. |
| **sefia_litellm** | `packages/sefia_litellm` | Provider adapter — an `LLMClient` implemented over LiteLLM. |
| **sefia_typer** | `packages/sefia_typer` | Typer (CLI) building blocks: the shared input channel with CLI reporting hooks, and the reporter surface. |
| **sefia_fastapi** | `packages/sefia_fastapi` | FastAPI (HTTP) building blocks: the shared input channel, SSE streams (`sefia_fastapi.events.SessionEvents`) with the SSE event names as the single source of truth (`SSEEvent`), and HTTP-facing exceptions in `sefia_fastapi.exceptions`. |
| **examples** | `examples` | Runnable end-to-end workflows. |
| **glyff** | *(separate repo)* | Content-addressed durable execution. A dependency, not vendored. |
| **jsonweir** | *(separate repo / PyPI)* | Standalone incremental JSON parser used for streaming tool args. |

## Package dependencies (one-way, no cycles)

Arrows point from a package to what it imports or depends on.

```
examples ─▶ sefios[all]

sefios ─┬▶ sefia
        ├▶ glyff / glyff-file-store / glyff-pydantic
        ├▶ sefia_litellm        (optional: sefios[litellm])
        ├▶ ddgs                 (optional: sefios[web])
        ├▶ sefia_typer          (optional: sefios[cli])
        └▶ sefia_fastapi        (optional: sefios[fastapi])

sefia_litellm ─┬▶ sefia
               ├▶ litellm
               └▶ jsonweir

sefia_typer ─┬▶ sefia
             └▶ typer / rich

sefia_fastapi ─┬▶ sefia
               └▶ fastapi

sefia ─┬▶ pydantic
       └▶ glyff / glyff-file-store / glyff-pydantic

jsonweir is a separate package published on PyPI, not a workspace member.
```

Rules that keep the layering clean — worth preserving in any change:

- **`sefia` core never imports `sefios`, `sefia_litellm`, or any provider SDK.**
  Provider, IO, and convenience concerns live above the core.
- **Provider specifics stay in adapters.** Anything LiteLLM/OpenAI-shaped belongs in
  `sefia_litellm`, behind the `LLMClient` interface. The core only knows the interface.
- **`sefios` depends on `sefia` directly.** Its provider, web, CLI, and HTTP
  integrations are optional extras, so installing `sefios` alone pulls in only the core.
- **Framework adapters depend only on `sefia` and their framework.** `sefia_typer`
  owns CLI reporting and `sefia_fastapi` owns HTTP event streaming. Neither imports
  `sefios` or the other adapter; shared session/input orchestration belongs to the
  `sefios` composition layer.
- **`sefios` is the composition layer for the adapters.** The extra-gated
  `sefios/cli` and `sefios/fastapi` facades are the only modules that import
  `sefia_typer` / `sefia_fastapi`; they wire the adapters to `SessionScope`,
  `Input`, `Output`, session storage, and cost accounting.
- **`examples` depend on the batteries.** They are consumers of the stack, not a layer
  that other packages should import.

## Inside `sefia` (the core)

Modules with a leading underscore are internal. The package root is the primary
authoring surface; categorized extension APIs may also live in named public
submodules such as `sefia.llm.exceptions` and `sefia.llm.transports`.

| Module | Responsibility | Key symbols |
| --- | --- | --- |
| `_authoring/` | Authoring API split by responsibility: domain ownership and runtime engraving, inference assembly, profile/policy selection, tool markers, and decorator metadata. | `Domain`, `concurrent`, `preview`, `policy`, `profile` |
| `_executor.py` | The step loop, middleware composition. | `InferenceExecutor` |
| `_tool_execution.py` | Executes a decision's tool-call batch (serial by default, `@concurrent` calls overlap). | `call_tools` |
| `inference.py` | Plain data: the decision/history types and the call descriptor, including the receiver/prompt-data split. | `FunctionInfo`, `Capability`, `ToolCallsDecision`, `ResultDecision` |
| `_session.py` | Wraps a `glyff.Session`, builds the strategy, installs the context. | `Session` |
| `_context.py` | The contextvar-scoped run state. | `SessionContext`, `get_context` |
| `_history.py` | The run's conversation history as pure in-memory state (loading/persistence/step-count live on the executor). | `StepHistory` |
| `history_storages/` | `HistoryStorage` implementations (default: history in the run's glyff metadata). | `GlyffHistoryStorage` |
| `_profiles.py` / `_authoring/metadata.py` | Per-call model selection and the decorator metadata store. | `Profile` |
| `_tool_system/` | The tool system split by responsibility: `roles.py` owns `Tools[...]` and decorator metadata, `entries.py` owns definitions and executable entries, and `registry.py` owns registration and collection contracts. | `ToolEntry`, `SignatureToolEntry`, `JsonSchemaToolEntry`, `ToolDefinition`, `ToolRegistry`, `ToolCollector`, `Tools` |
| `_tool_context.py` | The serving tool call's id and callable identity, bound around each `invoke` and read from a handler body. | `current_tool_call_id`, `current_tool_call_id_for` |
| `_introspection.py` | Sefia-agnostic reflection: annotation unwrapping, method/field scanning for classes and `Protocol`s. | `unwrap_annotation`, `declared_methods`, `declared_fields`, `is_protocol` |
| `tool_collectors/` | Collector implementations: default discovery (`Tools[...]`-granted fields of the call's receiver, declared-only; surface protocols on `self`), fixed pre-built tools, and composition. | `DefaultToolCollector`, `StaticToolCollector`, `CompositeToolCollector` |
| `event_system.py` / `events.py` | Observation seam: publisher + event types. | `EventPublisher` |
| `streaming.py` | The tool-arg streaming side channel (`preview`). | `ArgStream`, `StringDelta` |
| `llm/` | The **default** `InferenceStrategy`: `LLMClient` returns a normalized completion; transports decode it to decision data; `step_decision.py` validates that data; prompt renderers own text; `streaming.py` decodes incremental JSON; and `_strategy.py` coordinates repair. | `LLMInferenceStrategy`, `LLMClient`, `LLMCompletion`, `StructuredData`, `DecodedDecision`, `DecisionSpec`, `DecisionTransport`, `PromptRenderer` |
| `llm/transports/` | Transport contract and structured, prompted, and native protocols. Native orchestration, prompt/history conversion, result-tool construction, and decoding are separate internal modules. | `DecisionTransport`, `StructuredDecisionTransport`, `PromptedDecisionTransport`, `NativeDecisionTransport` |
| `pydantic/` | The default `ModelBackend`: callable inspection plus result JSON Schema generation and restoration. It does not know the logical step-decision shape. | `PydanticModelBackend` |
| `testing.py` | Public test doubles/helpers for testing sefia-based code (used by the workspace's own tests and available to applications). | `MockLLMClient`, `MemoryHistoryStorage`, `result_completion`, `tool_calls_completion`, `memory_session` |

### The seams (`_interfaces/`) — the extension ports

These ABCs are where you swap behavior without touching the core. Each has a default
implementation noted in parentheses.

| Interface | Swap to… | Default |
| --- | --- | --- |
| `InferenceStrategy` | replace the "brain" (a different prompting scheme, or non-LLM) | `llm/LLMInferenceStrategy` |
| `PromptRenderer` | change decision-prompt and tool-result text representation | `llm/MarkdownPromptRenderer` |
| `DecisionTransport` | change how a decision request is prompted, sent, and decoded; raise `sefia.llm.exceptions.DecisionDecodingError` when a completion cannot be decoded as a decision | `llm/transports/` |
| `LLMClient` (in `llm/_client.py`) | add an LLM provider; raise `sefia.llm.exceptions.LLMCompletionDecodingError` for received responses that cannot be represented safely | `sefia_litellm.LiteLLMClient` |
| `ModelBackend` | replace callable inspection and result schema generation/restoration together | `pydantic/PydanticModelBackend` |
| `ToolCollector` | a different tool-discovery rule | `DefaultToolCollector` |
| `Policy` + `InferenceMiddleware`/`StepMiddleware` | control: retries, caps, guards — build one-offs with `Policy(handlers=..., middleware=...)` or subclass | `sefios` middleware/policies |
| `HistoryStorage` | where a run's history is persisted (enables compaction) | `GlyffHistoryStorage` (glyff metadata) |

## Inside `sefios` (the batteries)

| Path | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports the everyday authoring surface (`domain`, `concurrent`, `preview`, `policy`, `profile`, `Tools`, `Policy`, `Profile`) so application code imports only from `sefios`. |
| `_domain.py` | Convenience constructor for an application-owned `sefia.Domain`. |
| `_glyff.py` | Owns Sefios' runtime domain and stable names for its engraved tools. |
| `_scope.py` | `SessionScope` — the configured front door that wires client + glyff + store + defaults. |
| `persistence.py` | Persistence providers for execution, session state, and the session registry; memory is the default, with optional SQLite and JSON-file alternatives. |
| `_input_channel.py` | Internal persisted routing between the `Input` tool and host-provided CLI/HTTP input. |
| `policies/` | `DefaultPolicy` (step cap, stagnation detection, HITL call composition). |
| `middleware/` | `_max_steps`, `_retry`, `_stagnation`, `_input`, `_compaction` — control-seam behaviors. |
| `history_storages/` | `SessionHistoryStorage` — an alternative `HistoryStorage` that keeps run history in the session storage (keyed by the run's `ExecutionId`) instead of glyff metadata. |
| `handlers/` | `_cost` — an observation-seam handler (cost accounting). |
| `tools/` | `input.py` (external input, pause-by-raise), `output.py` (agent-authored, non-blocking output), `web.py` (DuckDuckGo search). |
| `storage/` | Session-scoped persistence: the `SessionStorage` interface + memory, SQLite, and JSON-file implementations. |
| `sessions/` | Durable `SessionRegistry` implementations, local `ActiveSessionStore`, and the CLI-oriented `SessionManager` that composes them. |
| `cli/` | Gated on `sefios[cli]`: `_app.py` owns the `SefiaCLI` session facade, `_reporting.py` bridges tool/session events to reporter DTOs, and `_cost_reporter.py` adds cost output; the package re-exports the `sefia_typer` reporter surface. |
| `fastapi/` | Gated on `sefios[fastapi]`: the `SefiaHTTP` facade composing `sefia_fastapi` with `SessionScope`, `Input`, `Output`, and SSE lifecycle/delta streaming; integration exceptions live in `sefios.fastapi.exceptions`. |
| `_state_store.py` / `_session_state.py` | Typed `StateStore`; the session-state binding and its accessors (`get_state`'s type-keyed tier sits on top; `get_call_state_store` / `get_session_storage` are the tool-facing tier). |
| `state.py` | App-level state helpers: `StateRegistry`, `StateContainer`, `state`, `get_state`. |

## Inside `sefia_litellm` (the provider adapter)

| Path | Responsibility |
| --- | --- |
| `_client.py` | `LiteLLMClient` orchestration, runtime logging configuration, and LiteLLM exception mapping. |
| `_request.py` | Converts core messages, tools, and a logical decision spec into LiteLLM wire messages and kwargs; selects the adapter schema policy for each tool. |
| `_response.py` | Decodes one completed LiteLLM response into `LLMCompletion`, including tool-argument restoration, usage, cost, and structured decision data. |
| `_streaming.py` | Consumes LiteLLM streams, accumulates completion text, and dispatches text, reasoning, and structured-data callbacks before delegating the completed response to `_response.py`. |
| `_native_tool_stream.py` | Decodes LiteLLM native tool-call fragments into provider-neutral argument progress events. |
| `_schema/_structured_decision.py` | Builds the provider-compatible decision schema, including the object-root union envelope, and restores completed output and stream paths to the logical `DecisionSpec` shape. |
| `_schema/_policy.py` | Declares independent generated/user-defined schema policies, applies permitted corrections, and validates the shared strict-output constraints. |
| `_schema/_uniform_dictionary.py` | Defines uniform-dictionary entry-array encoding and decoding. |
| `_schema/_data_format.py` | Translates provider-neutral structured data to and from one prepared wire schema; it has no tool knowledge. |

## Where to change what

| Goal | Where |
| --- | --- |
| Add an LLM provider | implement `LLMClient`; mirror `packages/sefia_litellm/src/sefia_litellm/_client.py` |
| Change the logical step-decision shape or validation | `llm/step_decision.py` |
| Change Pydantic result schema generation or restoration | `pydantic/_result_format.py` |
| Change generic `$defs` import or `$ref` rewriting | `llm/json_schema/_composition.py` |
| Change LiteLLM's structured decision format | `packages/sefia_litellm/src/sefia_litellm/_schema/` |
| Add a built-in tool | `packages/sefios/src/sefios/tools/` |
| Add retry / step-cap / a guard | a `Policy` + `StepMiddleware`/`InferenceMiddleware` in `sefios/middleware/` |
| Observe runs (logging, tracing, cost) | a handler over `events.py`; see `sefios/handlers/_cost.py` |
| Add a persistence backend | implement `PersistenceProvider` so the glyff execution backend, `SessionStorage`, and `SessionRegistry` are selected together; reference `persistence.py` |
| Compact a run's conversation history | add `HistoryCompactor` (`sefios/middleware/_compaction.py`); to change where history lives, pass `history_storage=` to `SessionScope`/`Session` (seam: `HistoryStorage`) |
| Change shared input routing / persistence rules | `sefios/_input_channel.py` |
| Change CLI rendering / input callbacks | `packages/sefia_typer` |
| Change HTTP events / SSE | `packages/sefia_fastapi` |
| Change how CLI or HTTP apps are wired to sessions, tools, and cost | the facades in `sefios/cli/` / `sefios/fastapi/` |
| Change which methods are tools (the `Tools[...]` grant rule) | `tool_collectors/_default.py`, role alias in `_tool_system.py`, scanners in `_introspection.py` |
| Per-call model/policy switch | `Profile` + the `@profile` decorator |
| Support a new authoring type system | implement `ModelBackend`; reference `pydantic/_model_backend.py` and `_result_format.py` |
| Register a tool from a raw JSON Schema (no signature) | `JsonSchemaToolEntry` / `ToolRegistry.add_json_tool` in `_tool_system/` |
| Read the serving call's id inside a tool body | `current_tool_call_id` / `current_tool_call_id_for` in `_tool_context.py` |
| Install a whole tool-discovery rule for a run (e.g. client-defined tools) | pass `tool_collector=` to `SessionScope`/`SessionScope.session()`/`Session` (seam: `ToolCollector`) |
| Trace the runtime end to end | [how-it-works.md](./how-it-works.md) |

## Conventions

- **Underscore = internal import path.** A public class may be implemented in an
  underscore module and deliberately exposed through the package `__init__.py`; users
  import the class from that facade, not its implementation module.
- **The package root is the primary authoring surface, not an inventory of every public
  type.** Narrowly categorized APIs such as events and exceptions, plus low-level APIs
  intended mainly for authors of Sefia extension libraries, live in descriptively
  named submodules without a leading underscore and are not re-exported from the
  package root. Ordinary application extension points and configuration types may
  remain at the root (for example, `Policy`).
- **Import from `sefios`.** It re-exports the everyday authoring surface
  (`domain` / `concurrent` / `preview` / `policy` / `profile`, `Tools`,
  `Policy` / `Profile`), so application code
  touches one package. Low-level contracts intended mainly for extension-library
  authors come from their specialized `sefia` submodules instead.
- **Interfaces live in `_interfaces/`** as ABCs; concrete defaults live in feature
  folders (`llm/`, `pydantic/`, `tool_collectors/`). Interfaces that belong to the
  ordinary authoring surface may be selected into the root facade; low-level contracts
  for extension-library authors should have a dedicated public submodule instead.
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
