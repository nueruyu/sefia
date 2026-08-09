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
4. History is saved after each step, so on re-invocation the run loads it and
   continues from the unfinished step without re-entering the completed ones.

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
  (`inference.py`), held in-memory by `StepHistory` and persisted by the executor
  through a `HistoryStorage` seam — see "History storage and compaction" below.
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

An invalid reply (empty body, malformed JSON, schema violation, unknown tool) is
first **repaired in place**: the strategy appends the invalid output and the
validation error to the conversation as corrective feedback and asks again, up to
`max_repair_attempts` times (default 2; configurable on
`LLMInferenceStrategy` / `Session` / `SessionScope`). The repair exchange lives only
inside that one (engraved) step's messages — it never enters the step history, so an
invalid decision is never persisted. Only when the budget is spent does the
`InvalidInferenceResponseError` propagate as described below.

(Why the unified schema rather than native tool-calling, and the tradeoff it makes:
[tradeoffs.md](./tradeoffs.md).)

## Tools: discovery, schema, execution

**Discovery** (`DefaultToolCollector.collect`) is gated by the `Tools` role alias —
there is **no ambient authority**. Tools come from the **receiver** (`self`/`cls`)
of the `@infer` call; every other parameter is task data (tool dependencies are
expressed through classes, so a plain `@infer` function has no tools). The receiver
is read in one of two modes:

- **Unannotated receiver** — the instance's **class-level field declarations** are
  scanned (`__annotations__` across the MRO plus read-only `property` declarations —
  never `getattr` on the instance, so a third-party object's lazy getters are never
  triggered). A field is exposed **only if annotated with the `Tools` alias**:
  `_web: Tools[WebToolkit]`. An unmarked or undeclared field exposes nothing, and
  the instance's own methods are never exposed.
- **Surface-annotated receiver** — a `self` annotated with a `Protocol` replaces the
  class-body scan with that protocol's declarations, granted wholesale: its methods
  (`_`-prefixed included) become tools bound to the instance, and its field/property
  declarations expose their declared type's members. Annotating `self` is itself the
  opt-in, so the surface needs no marker and stays a plain interface. The allowlist
  has no hidden exceptions: a surface that declares the running `@infer` method
  exposes it to itself — recursion is a declared choice, and bounding it is runtime
  policy (a planned depth-cap middleware), not discovery's job.

Either way, the **exposed interface is the declared type** (`Tools[...]` /
`Optional` stripped): a concrete class contributes its public methods; a `Protocol`
exactly its declared members — including `_`-prefixed ones, since a protocol is an
explicit allowlist. `Tools[T]` is an `Annotated` alias, so checkers treat it as `T`,
the wrapped type stays a plain class or protocol, and it stacks with other
`Annotated` metadata.

Discovery is a **pure function of static declarations**: runtime values never widen
the surface, and resolution is per field and fail-closed — an unresolvable annotation
(a forward reference, a `TYPE_CHECKING`-only or locally-scoped name) is skipped with
a debug log. If the model doesn't see a held field's methods, the usual cause is a
missing `Tools[...]` on the field's class-level annotation.

Properties and other non-function descriptors are never tools. A service becomes
another agent's tool by being held in a `Tools[...]`-annotated field, which makes a
sub-agent's `@infer` method an ordinary tool.

This also means the bound object is a capability boundary. If a class has multiple
`@infer` methods, they share the tool surface collected from its granted fields —
unless a method selects its own surface with a `self:` annotation. Split services
(or annotate `self`) when different operations need different tools or different
write permissions.

Three layers carry a tool from code to the model, each with its own name: the
**granted method** is the tool (the callable reached through a `Tools[...]`
field); a **`ToolEntry`** is the runtime registry record that binds that name to
a schema source and an invocation strategy — what other frameworks call a "Tool
object"; and a **`ToolDefinition`** is the model-facing schema embedded into the
prompt. Model-facing prose keeps this loose on purpose: a prompt says "use the
`WebSearch` tool", naming the granted method the model actually calls rather than
the entry behind it.

**Schema** (`_strategy.py`): each entry produces a `ToolDefinition`
(`ToolEntry.definition()`), embedded as JSON in the system prompt — not sent as a
native tool spec. A
`SignatureToolEntry` reflects its definition from a callable's signature via the
inspector; its `schema_source` is the *interface* method (a `Protocol`'s own
docstring and signature when the field was narrowed that way), while the callable
that runs stays the concrete, bound implementation — the same callable unless a
`Protocol` narrowed the field. The model is told the *Protocol's* parameter names,
and the executor calls the concrete implementation with those same names as
keyword arguments, so a `Protocol` and the implementation it narrows must agree on
parameter names, not just behavior — nothing checks this at runtime, so a mismatch
surfaces as a tool-execution error on the first call rather than at discovery time.
A `JsonSchemaToolEntry` instead carries its parameters as a raw JSON Schema (no
signature to introspect) and passes that schema through verbatim.

**Execution** (`_tool_execution.py`, engraved through
`InferenceExecutor._call_tools`): each requested call is matched
in the registry and dispatched through `ToolEntry.invoke(arguments)`; sync or async
returns are normalized. For a `SignatureToolEntry` the decoded arguments are coerced to
the callable's declared types before the call; a `JsonSchemaToolEntry` forwards them to
its handler verbatim. A tool that
**raises `InputRequired` propagates** — that is the durable pause (see below)
— so it reaches your handler; any *other* tool exception is stringified into the
history and fed back to the model so it can recover and continue, rather than failing
the run.

A handler that needs the identity of the call it is serving reads
`sefia.current_tool_call_id()` — the serving `ToolCallRequest.id`, stable across
the call's pause and resume (its decision replays from history), so a
transport-backed tool needs no glyff-derived per-call key.
`sefia.current_tool_call_id_for(function)` returns that id only when `function`
is itself the dispatched tool. The built-in Input/Output methods use this stricter
query as their `interaction_id` and fail fast when called directly, including from
inside another dispatched tool; they do not mint or inherit a second identity.

An `@preview` handler receives that same id before its `ArgStream`:
`handler(tool_call_id, events)`. A step-scoped registry in the inference strategy
allocates one id per tool-call index; both the streaming router and decision builder
request the id from that registry. Preview deltas and the authoritative tool result
can therefore refer to the same interaction even though the preview runs before tool
execution.

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

The nesting (run ⊃ step ⊃ tool batch) makes replay granular. On resume the
executor loads the saved `HistorySnapshot` and continues at `completed_steps`,
so finished steps are **not re-entered** — only the interrupted step runs again
(see "History storage and compaction").

**Exceptions never poison a run.** glyff **never engraves an exception as a permanent
result**: any exception that escapes an engraved call leaves that call **resumable**
while the work that already completed stays committed, and the exception then
propagates normally. So no exception type changes glyff's durability: a transient
provider hiccup or a response that failed schema validation simply propagates and is
re-run on the next invocation (the strategy's in-step feedback repair and an in-loop
`Retrier` may retry it first); an input tool raises `InputRequired` to pause; an ordinary
bug raises and surfaces to
you. In every case the completed engraved steps are safe and the interrupted one runs
again on re-invocation. sefia's control-flow pauses subclass `PauseException`
(`sefia.exceptions`), which the executor propagates untouched instead of reporting as
a failure: `InputRequired` (a tool awaiting input) and the recoverable `InferenceError`
base are both pauses.

## History storage and compaction

History is durable state the executor owns. `StepHistory` (`_history.py`) is a
pure in-memory item list; the executor loads a `HistorySnapshot` from a
`HistoryStorage`, tracks `completed_steps`, and saves after each completed step.
The default `GlyffHistoryStorage` writes the snapshot into the run execution's
glyff metadata in its own transaction scope, so it **commits immediately** even
though a `Never`-returning run never finishes. A resume reloads the snapshot and
continues from `completed_steps`, never re-entering finished steps. (sefios'
`SessionHistoryStorage` is an alternative that stores it in the session storage.)

Each step is engraved on its **ordinal**, not the history contents, so the
durable key stays O(1) and compaction can reshape the history without changing
any step's key. `HistoryCompactor` (`sefios.middleware`) truncates the history
via `ctx.history.rewrite` once it grows past a threshold; the executor persists
that rewrite just before the model call, so a resume loads the compacted
snapshot and never re-runs the compactor (safe even for a non-deterministic
one).

## Human-in-the-loop: pause = raise, resume = re-invoke

An input tool (`packages/sefios/src/sefios/tools/input.py`) is an engraved tool
that:

1. looks up whether input is recorded; if so, returns it;
2. if not, records the pending prompt and **raises `InputRequired`**.

The raise propagates out, glyff leaves that engraved tool call **resumable**, and the
exception reaches your handler, which returns "needs input". On the next request the
input is delivered with `accept_input` and you re-invoke the same session. Its
saved history is restored and the interrupted input tool continues with the
provided value.

Before tool execution, the default `sefios` policy also runs a step middleware that
composes multiple input tool calls emitted in the same model decision into one
prompt. It does not carry state across steps, so a follow-up question produced
after resume remains a normal separate interaction.

The idempotency hinge is the `ToolCallRequest.id`. The input tool uses it as its
`interaction_id`; because a resumed invocation restores the same decision from
history, it routes the reply to the same pending prompt instead of creating a
duplicate. The input/output preview callbacks receive this id alongside each text
delta, so HTTP SSE clients can merge a live preview directly into its authoritative
`input_required` or `output` event.

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
3. The input tool finds no input, records the prompt under its call-scoped state,
   and raises `InputRequired`. The search step had already been recorded to the history
   snapshot; the ask-input step had not. glyff leaves the input call resumable, and
   the exception surfaces; the handler returns `needs_input` + the prompt.
4. `POST /turn` again with the input (delivered via `accept_input`). `service.run`
   re-enters: the executor loads the snapshot and resumes at the ask-input step —
   the completed search step is **not re-entered**. That step's model decision
   replays its stored output (the draft is identical), the input tool re-runs, now
   finds the input, and returns it; the loop continues to the final answer.

Nothing ran between the two requests; the only thing that crossed the gap was rows in
the store.

## See also

- [infer-contract.md](./infer-contract.md) — function shapes, arguments, service members, tools, and return types.
- [DESIGN.md](../DESIGN.md) — why these choices.
- [glyff](https://github.com/nueruyu/glyff) — the engrave/replay engine in detail.
