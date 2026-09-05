# Tradeoffs

sefia keeps the abstraction small: selected typed Python calls become LLM-backed,
durable, and replayable. That is a bet on an abstraction boundary, not a claim
that broader agent frameworks or workflow systems are unnecessary.

## The boundary

The core boundary is an ordinary typed Python function. The function signature,
return type, and docstring remain the interface; the model/tool execution
underneath becomes replayable.

This gives sefia a narrow shape: it is not a general workflow engine, graph
runtime, or multi-agent platform.

## What sefia optimizes for

- ordinary Python control flow around LLM steps;
- typed return values as the output contract;
- replayable model/tool outputs after restart;
- pause/resume by re-invocation over request/response handlers;
- provider-portable decisions.

### Replayable model and tool steps

A completed model or tool step replays its stored output instead of running again.
That matters beyond saving money: LLM calls are non-deterministic, so a resumed
run must continue from the same draft, tool result, or approval point that the
paused run saw.

### Typed calls as the interface

The Python return type is the output contract. The application calls a function
and receives a typed value; the LLM interaction remains behind that call boundary.

This keeps application code close to ordinary Python, but it also means the
function boundary is where durability and replay are expressed.

### Provider portability over native provider tooling

sefia does not use native provider tool-calling by default. It asks the model for
one decision shape: final answer or tool calls.

`NativeDecisionTransport` is also available for provider-native function calls,
and `@concurrent` enables overlapping safe tool calls with any transport.
Applications
that rely on native parallel tool calls, provider-specific tuning, or
provider-managed caching may fit better with a native-tooling framework.

## What sefia does not try to replace

sefia does not replace provider-native tool execution or a general workflow
runtime.

- it does not provide native provider tool-calling by default;
- it does not include a distributed workflow runtime;
- it does not wake paused runs by itself;
- it assumes deterministic orchestration between replayed calls;
- long-running or cross-service workflows need another layer.

## When to use another layer

Use a workflow or durable-execution engine directly when the workflow itself is
the main artifact: long timers, workers and queues, cross-service compensation,
distributed fan-out, or audit-grade workflow history.

For LLM applications that need both an agent framework and durable execution, the
closer comparison is an agent framework plus a supported durable-execution
backend, such as Pydantic AI + Temporal or Pydantic AI + DBOS.

For a per-need table, see [choosing.md](./choosing.md).
