# Tradeoffs

sefia keeps the abstraction small: typed Python functions, replayable model/tool
calls, and explicit stores. That is a bet on an abstraction boundary, with real costs.

## What it buys

- ordinary Python control flow around LLM steps;
- typed return values as the output contract (any nested/union/collection type);
- pause and resume by re-invocation, on plain request/response handlers;
- no workflow engine to operate for request-scoped flows;
- one provider-portable result shape, so provider tool-call differences don't surface
  in your code.

## What it costs

- no native provider tool-calling — one unified schema instead, so no native parallel
  tool calls and prompt caching becomes something to design for;
- no built-in distributed workflow runtime;
- no self-waking timers (a paused run resumes only when something calls it);
- replay assumes deterministic orchestration between engraved calls.

## When a workflow engine fits better

Reach for Temporal- or DBOS-grade infrastructure when you need:

- long-running or self-firing timers (days to weeks);
- cross-service transactions with compensation;
- distributed fan-out across machines;
- audit-grade, queryable workflow history.

For a per-need table, see [choosing.md](./choosing.md).
