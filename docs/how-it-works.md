# How it works

The mechanism behind `@infer`, with enough detail to check it against the source.
There is nothing unobservable here: it is a prompt, a loop, a JSON schema, and
content-addressed replay. Module paths point at the code so you can verify each claim.

> Pre-1.0: the architecture below is stable, but names and one rule (which methods
> count as tools) are being finalized — see the notes inline and [DESIGN.md](../DESIGN.md).

## The pieces

| Piece | File | Role |
| --- | --- | --- |
| `@infer` decorator | `packages/sefia/src/sefia/_decorators.py` | Wraps a function so calling it runs an inference instead of the body. |
| `InferenceExecutor` | `packages/sefia/src/sefia/_executor.py` | Owns the step loop, tool execution, middleware. |
| `LLMInferenceStrategy` | `packages/sefia/src/sefia/llm/_strategy.py` | Turns the function + history into a prompt + schema, parses the reply. |
| `DefaultToolCollector` | `packages/sefia/src/sefia/tool_collectors/_collector.py` | Discovers tools from the agent object and its held dependencies. |
| `Session` / `SessionContext` | `packages/sefia/src/sefia/_session.py`, `_context.py` | The durable, contextvar-scoped run; wraps a `glyff.Session`. |
| glyff | [nueruyu/glyff](https://github.com/nueruyu/glyff) | Content-addressed engrave/replay underneath every engraved call. |

**One sentence of data flow:** calling an `@infer` function resolves the session
context, builds an executor, and runs it inside an *engraved* boundary; the executor
loops — ask the strategy for the next decision, run any tool calls, append to history
— until the model returns a final answer, with every model call and tool batch
individually engraved so a re-invocation replays completed work instead of redoing it.

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
  re-run on resume.
- **History** is the accumulating list of `ToolCallDecision` / `ToolCallResult`
  (`inference.py`) that gets rendered back into messages each step.
- **Two seams, kept apart:** *middleware* wraps the loop and can retry or
  short-circuit (control); the *event publisher* emits `BeforeInferenceStep`,
  `AfterToolCall`, etc. for handlers that can only observe — the publisher isolates a
  handler's exceptions so observation can never change the outcome.

## Turning a function into a prompt

`LLMInferenceStrategy.decide_next_step` (`llm/_strategy.py`) does the core
translation. The key design choice is that **tool-calling is not native** — there is
one unified structured-output schema instead.

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

**Why this shape:** one schema that works across any provider's JSON/structured-output
mode, and a return type that can be any nested/union/collection type — at the cost of
native parallel tool calls and getting prompt caching for free. (See
[DESIGN.md](../DESIGN.md#non-goals--tradeoffs) and
[tradeoffs.md](./tradeoffs.md#2-provider-leakage).)

## Tools: discovery, schema, execution

**Discovery** (`DefaultToolCollector.collect`): given the agent instance (`self` of
the `@infer` method), the collector gathers tools from the instance itself and from
**each dependency it holds in an attribute** (public or private). It scans the class
hierarchy via `__mro__` (and `__slots__`) rather than `dir()`+`getattr` on every
name, so it never triggers a third-party object's lazy properties. Today a method
counts as a tool when it carries the `@tool` marker (or comes from a `toolify`
`Toolset`); **the rule being finalized is to treat the public methods of held objects
as the tool surface** (private `_` methods internal) — see DESIGN. Either way, the
*mechanism* is the same: collect from the held objects into a `ToolRegistry`. The
running `@infer` method is itself unmarked, so an agent never offers itself as a tool.

**Schema** (`_strategy.py`): each tool's signature becomes a function schema
(`model_inspector.get_function_schema`) and is embedded as JSON in the system prompt —
not sent as a native tool spec.

**Execution** (`InferenceExecutor._call_tools`): the model's requested call is matched
in the registry and invoked; sync or async returns are normalized. A tool that
**raises `NeedsInput` propagates immediately** — that is the durable pause (see below)
— so it reaches your handler; any *other* tool exception is stringified into the
history and fed back to the model so it can recover and continue, rather than failing
the run.

## Durability and replay (glyff)

Every engraved call (the `@infer` run, each model step, each tool batch) is keyed by
glyff on its **call identity + arguments** (content-addressed). On a later invocation
of the same session:

- a call that **already completed** returns its **stored output** instead of running
  again — so a non-deterministic model step yields the *same* result it did the first
  time (correctness, not just cost); and
- only the **unfinished** call actually executes.

The nesting (run ⊃ step ⊃ tool batch) makes replay granular: resuming a turn that
paused at step 4 replays steps 1–3 and the tools they called, then runs step 4.

**Exceptions never poison a run.** glyff **never engraves an exception as a permanent
result**: any exception that escapes an engraved call leaves that call **resumable**
while the work that already completed stays committed, and the exception then
propagates normally. So there is no special control-flow type: a transient
provider hiccup or a response that failed schema validation simply propagates and is
re-run on the next invocation (an in-loop `Retrier` may retry it first); a
human-input tool raises `NeedsInput` to pause; an ordinary bug raises and surfaces to
you. In every case the completed engraved steps are safe and the interrupted one runs
again on re-invocation.

> Pre-1.0: today's code still routes recoverable errors through a transitional
> exception base (`InferenceError` in `exceptions.py`); that distinction is being
> removed so *every* exception is treated as non-terminal, as described above.

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

The idempotency hinge is `get_call_state_store` (`_context.py`): it scopes a small
state store to the **current engraved call's `ExecutionId`** (hashed). Because a
resumed invocation re-enters the *same* engraved call with the *same* execution id,
the tool reads back the *same* `interaction_id` it stored before — so the pending
question is keyed stably and a re-entry doesn't create a duplicate. State that must
survive a pause lives here; everything else is just function arguments and return
values.

## Sessions and context

`sefia.Session` (`_session.py`) wraps a `glyff.Session`, builds the strategy/tool
collector/stores, and on `__aenter__` installs a `SessionContext` into a contextvar
that `@infer` reads. `SessionScope` in `sefios` is the configured front door that
constructs all of this (LLM client, glyff session, file store, default policies) so
application code only writes `async with scope.session(session_id=...)`. Profiles let
a single call swap the model/policies by key, resolved per-call in
`SessionContext.resolve_profile`.

## End to end: tracing one pause/resume

1. `POST /turn` → `scope.session(id)` installs the context → `agent.run(task)`.
2. `@infer` engraves the run; the executor loops: model step (engraved) → "search"
   tool call (engraved) → model step → "ask human to approve" tool call.
3. The human tool finds no answer, records the question under its call-scoped state,
   and raises `NeedsInput`. glyff keeps the run's completed steps, leaves the human
   call resumable, and the exception surfaces; the handler returns `needs_input` + the
   question.
4. `POST /turn` again with the answer (delivered via `accept_input`). `agent.run`
   re-enters: the search step and the
   earlier model steps **replay their stored outputs** (the draft is identical), the
   human tool re-runs, now finds the answer, and returns it; the loop continues to the
   final answer.

Nothing ran between the two requests; the only thing that crossed the gap was rows in
the store.

## See also

- [DESIGN.md](../DESIGN.md) — why these choices.
- [glyff](https://github.com/nueruyu/glyff) — the engrave/replay engine in detail.
