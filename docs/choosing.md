# Choosing a stack

sefia is narrow in the abstraction it adds: selected typed Python calls become
LLM-backed, durable, and replayable. It is not a general workflow engine, graph
runtime, or multi-agent platform — where one of those fits the shape better, use it.

Note the layers: Pydantic AI, LangGraph, CrewAI, and sefia are agent/orchestration
abstractions, while Temporal and DBOS are durable-execution engines. So the
apples-to-apples comparison to an agent-plus-durability setup is **Pydantic AI +
Temporal/DBOS**, not Temporal or DBOS on their own.

> Category-level and current as of writing; these tools move fast, so check each one.

## Quick table

| Need | Consider |
| --- | --- |
| One-shot typed model calls | a provider SDK, or Pydantic AI |
| Typed agent runtime with native provider tool-calling | Pydantic AI |
| A typed agent runtime plus a durable-execution backend | Pydantic AI + Temporal / DBOS |
| An explicit graph / state machine as the artifact | LangGraph |
| Multi-agent crews, roles, and task orchestration | CrewAI |
| Durable typed Python calls that pause/resume over request/response | **sefia** |

## When sefia fits

- your workflow reads naturally as a Python call graph;
- model/tool outputs must replay unchanged after a restart;
- a human may approve or supply input later;
- you don't want to operate a workflow engine for this flow.

## When another tool may fit better

- **Pydantic AI** — a mature typed agent runtime with native provider tool-calling
  (parallel calls, per-provider tuning) and a broad ecosystem. It is sefia's closest
  neighbor; the differences are *where durability lives* and *how directly your code
  uses provider-native tool-calling*. For restart-surviving durability, pair it with
  a backend such as **Temporal** or **DBOS**, or use its native deferred-tool patterns.
- **LangGraph** — when the graph / state machine is the artifact you want to inspect
  and operate, with a built-in checkpointer and interrupt.
- **CrewAI** — when the main abstraction is a team of agents with roles, tasks, and
  automation flows.
- **A workflow engine directly (Temporal / DBOS)** — when the durable workflow is the
  point and it isn't LLM-specific: long timers, distributed fan-out, cross-service
  compensation, audit-grade history. For agent applications, this often means pairing
  an agent framework with one of these backends.

## Rule of thumb

If the flow is ordinary Python that has to pause and resume, sefia keeps it that way.
If the flow *is* a graph, is multi-agent, spans services, or must wake itself on a
timer, reach for the tool built for that. The reasoning is in
[tradeoffs.md](./tradeoffs.md).
