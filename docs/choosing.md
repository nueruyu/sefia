# Choosing a stack

sefia is narrow in its abstraction — typed Python calls made durable and replayable —
not in what you can build with it. Where another tool fits the shape better, use it.

> Category-level and current as of writing; these tools move fast, so check each one.

## Quick table

| Need | Consider |
| --- | --- |
| One-shot typed model calls | a provider SDK, or Pydantic AI |
| Model-driven agent runtime with native tool-calling | Pydantic AI |
| An explicit graph / state machine as the artifact | LangGraph |
| General durable execution on Postgres | DBOS |
| Distributed workflows, long timers, cross-service compensation | Temporal |
| Durable typed Python calls that pause/resume over request/response | **sefia** |

## When sefia fits

- your workflow reads naturally as a Python call graph;
- model/tool outputs must replay unchanged after a restart;
- a human may approve or supply input later;
- you don't want to operate a workflow engine for this flow.
- you want ordinary functions and service classes, not an agent object model.

## When another tool may fit better

- **Pydantic AI** — a mature typed agent runtime with native provider tool-calling
  (parallel calls, per-provider tuning) and a broad ecosystem. It is sefia's closest
  neighbor; the difference is *how you get durability* (an engine or deferred tools vs.
  native to `@infer`), *whether provider tool-calling leaks* into your code, and
  whether you want an agent object model or ordinary Python services/functions.
- **LangGraph** — when the graph / state machine is the artifact you want to inspect
  and operate, with a built-in checkpointer and interrupt.
- **DBOS** — general durable execution when you're already comfortable running Postgres
  as the store; the LLM layer stays application code on top.
- **Temporal** — distributed workflows across machines, long self-firing timers,
  cross-service compensation, or audit-grade workflow history.

## Rule of thumb

If the flow is ordinary Python that has to pause and resume, sefia keeps it that way.
If the flow *is* a graph, spans services, or must wake itself on a timer, reach for an
engine. The reasoning is in [tradeoffs.md](./tradeoffs.md).
