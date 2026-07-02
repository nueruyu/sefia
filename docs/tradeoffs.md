# Tradeoffs

sefia keeps the abstraction small: typed Python functions, replayable model/tool
calls, and explicit stores. That is a bet on an abstraction boundary, not a claim
that broader workflow systems are unnecessary.

## What it buys

The main bet is that ordinary typed Python calls are a good boundary for durable LLM
work. The function signature, return type, and docstring remain the interface; the
model/tool execution underneath becomes replayable.

- ordinary Python control flow (branches, loops, retries) around LLM steps;
- typed return values as the output contract — any nested/union/collection type,
  validated on the way out;
- pause and resume by re-invocation, on plain request/response handlers;
- request/session-scoped durability without adopting a workflow engine;
- one provider-portable result shape.

### Replay is for correctness, not just cost

A completed model or tool step replays its stored output instead of running again.
That matters beyond saving money: LLM calls are non-deterministic, so if a human
approved one draft, a resume must continue from *that* draft, not a freshly generated
one. Replay makes the resumed run identical to the one that paused.

### Unified schema vs native provider tooling

sefia doesn't use native provider tool-calling by default; it asks the model for one
result shape (`final_answer | tool_calls`) with strict structured output where
supported. The upside is that the Python return type stays the primary output
contract, independent of each provider's tool-call message format, and one decision
model covers both final answers and tool calls. The downside is that applications
that rely on native parallel tool calls, provider-specific tuning, or provider-managed
caching may fit better with a native-tooling framework.

### Stateless HTTP follows from the model

Because nothing runs between requests — a paused run is just an engraved call that
hasn't finished — the same model resumes on a plain request/response handler with no
background worker. Human-in-the-loop over stateless HTTP is one use of that model,
not a separate runtime protocol.

## What it costs

These costs are mostly about choosing where orchestration lives. sefia keeps the
application shape small, but it does not replace provider-native tool execution or
a general workflow runtime.

- no native provider tool-calling by default (see above);
- no built-in distributed workflow runtime;
- no self-waking timers — a paused run resumes only when something calls it;
- replay assumes deterministic orchestration between engraved calls;
- long-running or cross-service workflows need another system.

## Layer boundary

Durable-execution systems such as Temporal and DBOS solve a broader workflow problem
than sefia does.

For LLM applications, the closer comparison is often an agent framework plus a
supported durable-execution backend, such as Pydantic AI + Temporal or Pydantic AI +
DBOS.

sefia makes a different tradeoff: replay is part of the typed function model itself.
The application still looks like ordinary Python functions/classes, while selected
calls become LLM-backed and replayable.

## When a workflow engine fits better

Use a workflow or durable-execution engine directly when you need:

- long-running or self-firing timers (days to weeks);
- workers and queues as core infrastructure;
- cross-service transactions with compensation;
- distributed fan-out across machines;
- audit-grade, queryable workflow history.

For a per-need table, see [choosing.md](./choosing.md).
