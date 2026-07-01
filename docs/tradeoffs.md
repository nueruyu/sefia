# Tradeoffs

sefia keeps the abstraction small: typed Python functions, replayable model/tool
calls, and explicit stores. The bet is that ordinary typed Python calls are a good
boundary for durable LLM work — the signature, return type, and docstring are the
interface, and the model/tool execution underneath is replayable.

## What it buys

- ordinary Python control flow (branches, loops, retries) around LLM steps;
- typed return values as the output contract — any nested/union/collection type,
  validated on the way out;
- pause and resume by re-invocation, on plain request/response handlers;
- no workflow engine to operate for request-scoped flows.

### Replay is for correctness, not just cost

A completed model or tool step replays its stored output instead of running again.
That matters beyond saving money: LLM calls are non-deterministic, so if a human
approved one draft, a resume must continue from *that* draft, not a freshly generated
one. Replay makes the resumed run identical to the one that paused.

### The unified schema

sefia doesn't use native provider tool-calling by default; it asks the model for one
result shape (`final_answer | tool_calls`) with strict structured output where
supported. The upside is that the Python return type stays the primary output
contract, independent of each provider's tool-call message format, and one decision
model covers both final answers and tool calls. The downside is losing native parallel
tool calls and some provider-specific optimizations (prompt caching becomes something
to design for).

### Stateless HTTP is a consequence, not a feature

Because nothing runs between requests — a paused run is just an engraved call that
hasn't finished — the same model resumes on a plain request/response handler with no
background worker. Human-in-the-loop over stateless HTTP falls out of the design; it
isn't bolted on.

## What it costs

- no native provider tool-calling (see above);
- no built-in distributed workflow runtime;
- no self-waking timers — a paused run resumes only when something calls it;
- replay assumes deterministic orchestration between engraved calls.

## When a workflow engine fits better

Reach for Temporal- or DBOS-grade infrastructure when you need:

- long-running or self-firing timers (days to weeks);
- cross-service transactions with compensation;
- distributed fan-out across machines;
- audit-grade, queryable workflow history.

For a per-need table, see [choosing.md](./choosing.md).
