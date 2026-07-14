# How it works

This page traces the runtime path behind `@infer` and points to the modules that
implement each step.

> Pre-1.0: some names and the tool-discovery rule may change.

## The pieces

| Piece | File | Role |
| --- | --- | --- |
| `@infer` decorator | `packages/sefia/src/sefia/_decorators.py` | Wraps a function so calling it runs an inference instead of the body. |
| `InferenceExecutor` | `packages/sefia/src/sefia/_executor.py` | Owns the step loop, tool execution, middleware. |
| `LLMInferenceStrategy` | `packages/sefia/src/sefia/llm/_strategy.py` | Turns the function + history into a prompt + schema, parses the reply. |
| `DefaultToolCollector` | `packages/sefia/src/sefia/tool_collectors/_default.py` | Discovers tools from the bound object and its held dependencies. |
| `Session` / `SessionContext` | `packages/sefia/src/sefia/_session.py`, `_context.py` | The durable, contextvar-scoped run; wraps a `glyff.Session`. |
| glyff | [nueruyu/glyff](https://github.com/nueruyu/glyff) | Content-addressed engrave/replay underneath every engraved call. |

At a high level:

1. Calling an `@infer` function resolves the current session context and builds an
   executor, run inside an *engraved* boundary.
2. The executor loops, asking the strategy for the next model decision.
3. Each model step and tool batch is engraved separately.
4. On re-invocation, completed work replays and the run continues from the unfinished
   step.

## `@infer`: calling a function runs an inference

`infer()` returns a wrapper that, on each call (`_decorators.py`):

1. **Resolves configuration** — gets the current `SessionContext` (a contextvar),
   reads the function's `__sefia_metadata__` for any `@policy`/`@profile`, resolves
   the profile to an inference strategy, and layers policies
   **session → profile → function** (most specific last). Policies produce *handlers*
   (observation) and *middleware* (control), which are split into the two seams.
2. **Builds an `InferenceExecutor`** with the function, the bound args, the strategy,
   and the tool collector.
3. **Engraves the run.** The configuration above happens *outside* the engrave
   boundary on purpose — a misconfigured policy should surface as an ordinary error,
   not an engraved failure that replays forever. Only `executor.run()` is wrapped in
   an inner `@engrave` that takes the user's args, so glyff keys the durable record
   on the call and its arguments.

The original function body is never executed — it exists only so its **signature,
type hints, and docstring** can be read (`FunctionInfo.create` in `inference.py`).
For user-facing constraints on function shapes, arguments, service members, tools,
and return types, see [infer-contract.md](./infer-contract.md).

## The inference loop

`InferenceExecutor._attempt_inference` (`_executor.py`) is a plain `while True`:

```
loop:
  decision = strategy.decide_next_step(function_info, history, tools)   # one model call
  if decision is FinalAnswer:  return decision.answer
  if decision is ToolCalls:    history += decision; history += run(decision.calls)
```

- **`decide_next_step` is engraved** (`_next_step_engraved`), so each model call is a
  separately replayable step.
- **The tool batch is engraved** (`_call_tools_engraved`), so executed tools don't
  re-run on resume. Within a batch, calls run serially unless their tools are
  marked `@concurrent`; results always land in history in request order (see
  [Tools](#tools-discovery-schema-execution)).
- **History** is the accumulating list of `ToolCallDecision` / `ToolCallResult`
  (`inference.py`) that gets rendered back into messages each step.
- **Two seams, kept apart:** *middleware* wraps the loop and can retry or
  short-circuit (control); the *event publisher* emits `BeforeInferenceStep`,
  `AfterToolCall`, etc. for handlers that can only observe — the publisher isolates a
  handler's exceptions so observation can never change the outcome.

## Turning a function into a prompt

`LLMInferenceStrategy.decide_next_step` (`llm/_strategy.py`) does the core
translation. Instead of provider-native tool-calling, it asks the model for one
unified structured-output schema.

A `create_model`-built decision model is the schema, picked by an
`_ExecutionDirector`:

| Return type | Director | Decision schema |
| --- | --- | --- |
| `Never` | `_ToolOnlyDirector` | `{ tool_calls }` only — must keep calling tools, no final answer |
| has tools | `_ToolEnabledDirector` | `{ final_answer: T \| null, tool_calls: [...] \| null }`, exactly one non-null |
| no tools | `_OutputOnlyDirector` | `{ final_answer: T }` only |

The system prompt is `docstring + response-instructions + the tool definitions (as
JSON) + the decision JSON Schema`. The user message is the call's arguments rendered
as XML (`_build_messages`); prior steps are replayed as ordinary
assistant/tool messages. The client is always called with `tools=None` and the
unified `output_schema` — provider native tool-calling is never used. The reply is
stripped of any ``` fence, `json.loads`-ed, validated into the decision model, and
`process_decision` validates `final_answer` against the declared return type
(`InvalidInferenceResponseError` if it doesn't conform).

(Why the unified schema rather than native tool-calling, and the tradeoff it makes:
[tradeoffs.md](./tradeoffs.md).)

## Tools: discovery, schema, execution

**Discovery** (`DefaultToolCollector.collect`): tools are the **public surface of
what the bound instance (`self` of the `@infer` method) holds** — plain Python
visibility, no marker or registry. For **each dependency held in an attribute**
(public or private; found via `__mro__`/`__slots__`, not `dir()`+`getattr` on every
name, so a third-party object's lazy properties are never triggered), the collector
resolves an *interface* for that field and exposes its public (non `_`-prefixed)
methods:

- If the field has a **class-level annotation** (`_web: WebToolkit`, `_web:
  ReadOnlyWeb`, assigned in the class body — not just an `__init__` parameter, whose
  mapping to the attribute is unrecoverable), that declared type is the interface: a
  concrete class exposes its public methods, a `Protocol` exposes only its declared
  members. `Any`/`object` declare no interface and behave as if unannotated.
- Otherwise the interface falls back to the **runtime value's concrete type**.

Narrowing is **best-effort and fails open**: the annotation is resolved with
`typing.get_type_hints`, so a type it cannot resolve — most commonly a `Protocol`
or class defined in a local scope (inside a function, e.g. a test), which
`get_type_hints` cannot see — silently falls back to the runtime type and exposes
the **full** public surface. If you use a `Protocol` to *restrict* a broad object's
surface (hiding a destructive method), declare that `Protocol` at module level, not
locally, or the restriction is silently lost.

Properties and other non-function descriptors are never treated as tools (accessing
them could execute a getter's side effects). The instance's **own** methods —
including its `@infer` methods — are never offered back to itself as tools; that
dissolves self-recursion entirely. A service becomes usable as a tool by being
*held* as another service's dependency, not by marking its own methods — so a
sub-agent's `@infer` method is a normal, callable tool once something else holds it.

This also means the bound object is a capability boundary. If a class has multiple
`@infer` methods, they share the tool surface collected from that instance and its
held dependencies. Keep multiple inferred methods together only when that shared
surface is intentional; split services when different operations need different
tools or different write permissions.

**Schema** (`_strategy.py`): each tool produces a `ToolDefinition` (`tool.definition()`),
embedded as JSON in the system prompt — not sent as a native tool spec. A
`SignatureTool` reflects its definition from a callable's signature via the
inspector; its `schema_source` is the *interface* method (a `Protocol`'s own
docstring and signature when the field was narrowed that way), while the callable
that runs stays the concrete, bound implementation — the same callable unless a
`Protocol` narrowed the field. The model is told the *Protocol's* parameter names,
and the executor calls the concrete implementation with those same names as
keyword arguments, so a `Protocol` and the implementation it narrows must agree on
parameter names, not just behavior — nothing checks this at runtime, so a mismatch
surfaces as a tool-execution error on the first call rather than at discovery time.
A `JsonSchemaTool` instead carries its parameters as a raw JSON Schema (no
signature to introspect) and passes that schema through verbatim.

**Execution** (`_tool_execution.py`, engraved through
`InferenceExecutor._call_tools`): each requested call is matched
in the registry and dispatched through `tool.invoke(arguments)`; sync or async
returns are normalized. For a `SignatureTool` the decoded arguments are coerced to
the callable's declared types before the call; a `JsonSchemaTool` forwards them to
its handler verbatim. A tool that
**raises `NeedsInput` propagates** — that is the durable pause (see below)
— so it reaches your handler; any *other* tool exception is stringified into the
history and fed back to the model so it can recover and continue, rather than failing
the run.

When one decision contains several calls, the batch runs **serially by
default**; consecutive calls to `@concurrent`-marked tools overlap, and an
unmarked call is a barrier. This is not fire-and-forget: results are awaited
and appended to history **in request order** regardless of completion order,
identical calls never race (glyff sequences a content key by arrival), and a
pause lets overlapped siblings finish — an engraved sibling's work commits and
replays on resume — before the earliest pause in request order propagates.
Mark only tools that tolerate overlapping; leave tools unmarked when their
side-effect ordering matters or they mutate shared state without locking.

## Durability and replay (glyff)

Every engraved call (the `@infer` run, each model step, each tool batch) is keyed by
glyff on its **call identity + arguments** (content-addressed). For method calls,
those arguments include `self`: `self` is not prompt input, but it still contributes
to the durable execution identity. On a later invocation of the same session:

- a call that **already completed** returns its **stored output** instead of running
  again — so a non-deterministic model step yields the *same* result it did the first
  time (correctness, not just cost); and
- only the **unfinished** call actually executes.

The nesting (run ⊃ step ⊃ tool batch) makes replay granular: resuming a turn that
paused at step 4 replays steps 1–3 and the tools they called, then runs step 4.

**Exceptions never poison a run.** glyff **never engraves an exception as a permanent
result**: any exception that escapes an engraved call leaves that call **resumable**
while the work that already completed stays committed, and the exception then
propagates normally. So no exception type changes glyff's durability: a transient
provider hiccup or a response that failed schema validation simply propagates and is
re-run on the next invocation (an in-loop `Retrier` may retry it first); a
human-input tool raises `NeedsInput` to pause; an ordinary bug raises and surfaces to
you. In every case the completed engraved steps are safe and the interrupted one runs
again on re-invocation. sefia's control-flow pauses subclass `PauseException`
(`sefia.exceptions`), which the executor propagates untouched instead of reporting as
a failure: `NeedsInput` (a tool awaiting input) and the recoverable `InferenceError`
base are both pauses.

## Human-in-the-loop: pause = raise, resume = re-invoke

A human-input tool (`packages/sefios/src/sefios/tools/human.py`) is an engraved tool
that:

1. looks up whether an answer is recorded; if so, returns it;
2. if not, records the pending question and **raises `NeedsInput`**.

The raise propagates out, glyff leaves that engraved tool call **resumable**, and the
exception reaches your handler, which returns "needs input". On the next request the
answer is delivered with `accept_input` and you re-invoke the same session: every
completed step replays, and the human tool runs again, now with an answer available,
and returns it.

Before tool execution, the default `sefios` policy also runs a step middleware that
composes multiple human-input tool calls emitted in the same model decision into one
question. It does not carry state across steps, so a follow-up question produced
after resume remains a normal separate interaction.

The idempotency hinge is `get_call_state_store`
(`sefios/_session_state.py`): it scopes a small state store to the **current engraved
call's `ExecutionId`** (hashed). Because a resumed invocation re-enters the *same*
engraved call with the *same* execution id, the tool reads back the *same*
`interaction_id` it stored before — so the pending question is keyed stably and a
re-entry doesn't create a duplicate. The store commits immediately, so this state
survives the pause; everything else is just function arguments and return values.

## Sessions and context

`sefia.Session` (`_session.py`) wraps a `glyff.Session`, builds the strategy/tool
collector, and on `__aenter__` installs a `SessionContext` into a contextvar that
`@infer` reads. Session-scoped state persistence lives one layer up: `SessionScope` in
`sefios` is the configured front door that constructs all of this (LLM client, glyff
session, the session's `SessionStorage`, default policies) so
application code only writes `async with scope.session(session_id=...)`. Profiles let
a single call swap the model/policies by key, resolved per-call in
`SessionContext.resolve_profile`.

## End to end: tracing one pause/resume

1. `POST /turn` → `scope.session(id)` installs the context → `service.run(task)`.
2. `@infer` engraves the run; the executor loops: model step (engraved) → "search"
   tool call (engraved) → model step → "ask human to approve" tool call.
3. The human tool finds no answer, records the question under its call-scoped state,
   and raises `NeedsInput`. glyff keeps the run's completed steps, leaves the human
   call resumable, and the exception surfaces; the handler returns `needs_input` + the
   question.
4. `POST /turn` again with the answer (delivered via `accept_input`). `service.run`
   re-enters: the search step and the
   earlier model steps **replay their stored outputs** (the draft is identical), the
   human tool re-runs, now finds the answer, and returns it; the loop continues to the
   final answer.

Nothing ran between the two requests; the only thing that crossed the gap was rows in
the store.

## See also

- [infer-contract.md](./infer-contract.md) — function shapes, arguments, service members, tools, and return types.
- [DESIGN.md](../DESIGN.md) — why these choices.
- [glyff](https://github.com/nueruyu/glyff) — the engrave/replay engine in detail.
