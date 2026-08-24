# FAQ

Answers to the questions sefia tends to raise. Where the answer is a tradeoff, it's
stated as one.

## Positioning

### Isn't this just another LangChain-style agent framework?

There is no `Agent` object, chain, graph, or global tool registry. The unit of work
is a **typed async function** (`@infer`), composed with ordinary `await` and ordinary
Python control flow. Service classes can hold dependencies, and the public methods of
a field granted with the `Tools[...]` annotation become tools. If your logic is
naturally a Python call graph, there is no framework shape to adopt; you write
functions and classes.

### How is it different from Pydantic AI?

The closest neighbor on typed ergonomics, and a good library. Two differences: **how
you get durability** (Pydantic AI via native deferred tools or an adopted
`TemporalAgent`/`DBOSAgent` engine; sefia native to `@infer`, no engine) and **provider
coupling** (Pydantic AI binds to per-provider native tool-calling; sefia uses one
unified schema, at the cost below). The full comparison is in
[tradeoffs.md](./tradeoffs.md) and [choosing.md](./choosing.md).

### How is it different from LangGraph?

LangGraph models the flow as an explicit graph (nodes/edges/state) with a built-in
checkpointer and interrupt — strong when the flow *is* a state machine you
want to inspect and operate. sefia keeps the flow as a normal Python call graph and
makes the durability native instead of authoring a graph. Pick LangGraph when the
diagram is the artifact; sefia when it's ordinary code.

### How does resumption differ from Temporal or DBOS?

All three are checkpoint-and-replay at heart. The difference is **where the paused
run lives and what you operate**: Temporal keeps a suspended workflow in a
cluster + workers; DBOS keeps a background workflow in your process plus a mandatory
Postgres; sefia keeps **nothing** running between requests — state is in a store and
the next request replays. That makes sefia lighter and stateless-HTTP-native, and
makes it *unable to wake itself* (see timers, below). See
[tradeoffs.md](./tradeoffs.md).

### When should I *not* use sefia?

Long-horizon autonomous waits (days/weeks on a self-firing timer), cross-service
sagas with compensation, distributed fan-out of a single workflow across machines,
or audit-grade exactly-once history as a product requirement. Those justify a real
workflow engine. Multi-agent role-play orchestration is also out of scope. The full
boundary is in [tradeoffs — when you need the engine](./tradeoffs.md#when-a-workflow-engine-fits-better);
the decision guide is [choosing.md](./choosing.md).

## Mechanics

### How does the human-in-the-loop pause actually work? Exceptions?

Yes. An input tool checks for recorded input; if there isn't one, it records
the prompt and **raises `InputRequired`**. glyff treats exceptions
as non-terminal: completed engraved calls commit, the interrupted call stays resumable,
and the exception propagates so your handler can return "needs input". When the input
arrives in a later request (delivered with `accept_input`), you re-invoke the same
call; the completed steps replay their exact outputs and only the pending step runs. No
durable-execution engine, no websocket, no worker.

### What does "replay" guarantee, and why does it matter for correctness?

Each engraved call is content-addressed (call identity + arguments). On
re-invocation, a call that already completed returns its **stored output** instead of
re-running. This is not only a cost optimization: an LLM call is non-deterministic,
so re-running a `draft` step would produce a *different* draft than the human
approved. Replay returns the same draft.

### Do side effects fire exactly once?

A completed engraved step replays rather than re-executes, so its effect isn't
repeated. For an effect that crashes *after* doing the work but *before* it commits,
use an idempotency key at the boundary (the downstream dedupes) — the same discipline
any exactly-once system needs, but localized to that one step rather than spread
across hand-rolled bookkeeping.

### How do tools work — really just public methods?

Almost: a held dependency's **public methods are its tools when the field is granted
with the `Tools[...]` annotation** (`_web: Tools[WebToolkit]`); private
(`_`-prefixed) methods stay internal. `Tools[T]` is an `Annotated` alias — checkers
see plain `T`, the type stays an ordinary class — so this is one annotation on top of
ordinary OOP visibility: no decorator, no registry, no base class, and no ambient
authority (a held config or store never leaks as a tool). The grant must be a
class-level annotation (a bare class-body annotation is enough); discovery is
fail-closed. To
expose a narrower surface, grant through a `Protocol` (`Tools[ReadOnlyWeb]`), or
select a single method's tools by annotating its `self` with a plain surface
`Protocol`.

### mypy flags my `@infer` methods as `[empty-body]`

An `@infer` body is `...` by design (the LLM is the implementation), which mypy
reports as a missing return. Disable that one check where your agents live — per
module with `# mypy: disable-error-code="empty-body"`, or in config with
`disable_error_code = empty-body`. Do **not** reach for `@abstractmethod`: it silences
the message but makes the class abstract, so mypy then rejects instantiating it.
pyright does not raise this at all.

### What can I not do — the determinism constraint?

Replay assumes the orchestration **between** engraved calls takes the same path on
re-execution. Keep nondeterministic or side-effecting work *inside* engraved calls
(so the result is replayed, not recomputed) and keep the code around them
deterministic. This is every replay engine's caveat; sefia's is no stricter.

### What about long-running waits / timers?

Because nothing runs between requests, sefia cannot fire its own timer. A
"continue in 3 days" needs an **external trigger** — a cron, scheduler, or
delay-queue that calls the resume endpoint. Re-invocation is idempotent (replay makes
a repeated call safe), so this is a thin, debuggable part, and most stacks already
run a scheduler. If you need many self-firing long timers as the core of the
workload, that's the boundary where an engine is the right call.

## Constraints & production

### Why isn't native tool-calling the default?

The unified schema is a design choice, not only a portability workaround. Asking the
model for one result shape (`final_answer | tool_calls`) with strict structured output
lets sefia treat the **Python return type as the primary output contract** — any
nested/union/collection type — independent of a provider's native tool-call format.
The cost is real: some frontier-model tuning and caching advantages favor native
tools. `sefia_litellm.NativeToolCallTransport` is available for that tradeoff, and
`NativeResultTransport` can represent the typed final value as a synthetic result
tool while keeping the same internal decision model and durable history. Full treatment:
[tradeoffs.md](./tradeoffs.md).

### Does it scale?

Horizontally across independent sessions, as long as the backing store and session
locking are designed for it: because no instance *owns* a paused run, any instance can
resume any session, with no affinity to preserve. What sefia does *not* do is
distribute a single workflow across machines; that's an engine's job. Per-resume cost is replaying a session's completed steps, which is
cheap for typical turn-length histories and is the thing to watch for very long
single runs.

### Do I need Postgres?

No. `sefios.SessionScope` uses process-local memory by default, with no database
service or durable-backend dependency. Install `sefios[sqlite]` and select
`SQLitePersistence` for restart-safe local persistence. The optional
`FilePersistence` is intended for inspecting JSON during debugging. A
Postgres or other backend can implement the typed `PersistenceProvider` seam,
including a shared session registry. CLI active-session selection remains local
workspace state. Your application database stays yours; Sefia stores only enough to
bring a paused or crashed run back to where it was.

### Is it production-ready?

Pre-1.0. The API is unstable and parts of the design (notably the tool model) are
being finalized. Use it where you can track breaking changes; see
[DESIGN.md](../DESIGN.md) and the issue tracker for what is settled and what is in
flight.

## Packaging

### What are sefia, sefios, sefia_litellm, and glyff?

- **`sefia`** — the core: `@infer`, the tool model, sessions, durability glue.
- **`sefios`** — the opinionated batteries: the `SessionScope` front door, default
  policies/middleware, and tools (external input, web search).
- **`sefia_litellm`** — provider support via [LiteLLM](https://github.com/BerriAI/litellm)
  (installed by the `sefios[litellm]` extra).
- **[glyff](https://github.com/nueruyu/glyff)** — the content-addressed durable
  execution engine underneath; usable on its own, not LLM-specific.
