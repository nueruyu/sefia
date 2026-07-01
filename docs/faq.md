# FAQ

Answers to the questions sefia tends to raise. Where the answer is a
tradeoff, it's stated as one. For the longer arguments, see
[DESIGN.md](../DESIGN.md), [why-less.md](./why-less.md), and
[choosing.md](./choosing.md).

## Positioning

### Isn't this just another LangChain-style agent framework?

There is no `Agent` object, chain, graph, or global tool registry. The unit of work
is a **typed async function** (`@infer`), composed with ordinary `await` and ordinary
Python control flow. Tools are the public methods of the objects a function holds. If
your logic is naturally a Python call graph, there is no framework shape to adopt; you
write functions.

### How is it different from Pydantic AI?

It's the closest neighbor on typed ergonomics, and a good library. Two real
differences:

- **How you get durability.** Pydantic AI reaches durable HITL either by adopting an
  engine (its first-class `TemporalAgent`/`DBOSAgent` wrappers) or via its native
  *deferred-tools* flow, where a run returns, you persist the message history, and a
  new run resumes by passing the result back. sefia's durability is native to the
  `@infer` model (pausing is just raising, resuming is re-invoking) and runs on a
  plain handler with a sqlite/file/Postgres store, no engine.
- **Provider leakage.** Pydantic AI binds to per-provider native tool-calling, so
  provider differences (strict vs. best-effort structured output, parallel-call
  semantics) surface as caveats on your code. sefia uses one unified result shape, so
  the abstraction reads the same across providers — at the cost described below.

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
[why-less.md](./why-less.md#where-the-paused-run-lives--and-who-shares-stateless-over-http).

### When should I *not* use sefia?

Long-horizon autonomous waits (days/weeks on a self-firing timer), cross-service
sagas with compensation, distributed fan-out of a single workflow across machines,
or audit-grade exactly-once history as a product requirement. Those justify a real
workflow engine. Multi-agent role-play orchestration is also out of scope. The full
boundary is in [why-less — when you need the engine](./why-less.md#when-you-need-the-engine);
the decision guide is [choosing.md](./choosing.md).

## Mechanics

### How does the human-in-the-loop pause actually work? Exceptions?

Yes. A human-input tool checks for a recorded answer; if there isn't one, it records
the question and **raises `NeedsInput`**. glyff treats exceptions
as non-terminal: completed engraved calls commit, the interrupted call stays resumable,
and the exception propagates so your handler can return "needs input". When the answer
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

A held dependency's **public methods are its tools**; private (`_`-prefixed) methods
stay internal. Discovery follows ordinary OOP visibility on the objects an agent
holds — no decorator, no registry. To expose a narrower surface than a class's full
public API, hold it behind a `Protocol`: only the protocol's declared members are
offered. By convention an agent holds *only* its tools (so it never offers itself as
a tool); this is not statically enforced today.

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

### Why not native tool-calling? Don't I lose parallel tools and caching?

You do, and it's a deliberate bet. One unified result shape (`final_answer |
tool_calls`) plus strict structured output where supported buys
**provider-portability** and **full return-type expressiveness** with no per-provider
semantics leaking into your code. The cost is no native parallel tool calls and some
frontier-model tuning on long agent loops, and prompt caching becomes something to
design for rather than get for free. Concurrency and caching are tracked on the issue
tracker, not guaranteed. If you target one provider and lean on native parallel
tools, that calculus can flip. Full treatment:
[why-less — provider leakage](./why-less.md#2-provider-leakage).

### Does it scale?

Horizontally across independent sessions, as long as the backing store and session
locking are designed for it: because no instance *owns* a paused run, any instance can
resume any session, with no affinity to preserve. What sefia does *not* do is
distribute a single workflow across machines; that's an engine's job. Per-resume cost is replaying a session's completed steps, which is
cheap for typical turn-length histories and is the thing to watch for very long
single runs.

### Do I need Postgres?

No. The store is pluggable — memory and file stores ship; a Postgres or other backend
is an option, not a requirement. Your application database stays yours; sefia's store
holds only enough to bring a paused or crashed run back to where it was.

### Is it production-ready?

Pre-1.0. The API is unstable and parts of the design (notably the tool model) are
being finalized. Use it where you can track breaking changes; see
[DESIGN.md](../DESIGN.md) and the issue tracker for what is settled and what is in
flight.

## Packaging

### What are sefia, sefios, sefia_litellm, and glyff?

- **`sefia`** — the core: `@infer`, the tool model, sessions, durability glue.
- **`sefios`** — the official batteries: the `SessionScope` front door, default
  policies/middleware, and tools (human input, web search).
- **`sefia_litellm`** — provider support via [LiteLLM](https://github.com/BerriAI/litellm)
  (installed by the `sefios[litellm]` extra).
- **[glyff](https://github.com/nueruyu/glyff)** — the content-addressed durable
  execution engine underneath; usable on its own, not LLM-specific.
