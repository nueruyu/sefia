# Choosing a stack

sefia is narrow in the abstraction it adds: selected typed Python calls become
LLM-backed, durable, and replayable. It is not a general workflow engine, graph
runtime, or multi-agent platform.

> Category-level and current as of writing; these tools move fast, so check each one.

## Agent / orchestration layer

| Need | Consider |
| --- | --- |
| One-shot model calls or minimal typed calls | Provider SDK or Pydantic AI |
| Typed agent runtime with native provider tooling | Pydantic AI |
| Typed agent runtime with durable workflow backend | Pydantic AI + Temporal / DBOS |
| Explicit graph or state machine as the artifact | LangGraph |
| Multi-agent teams, roles, tasks, and flows | CrewAI |
| Ordinary typed Python functions/classes with replayable LLM/tool steps | **sefia** |

## Durability / workflow layer

| Need | Consider |
| --- | --- |
| Long timers, workers, distributed workflows, compensation | Temporal |
| Durable Python workflows backed by a database | DBOS |
| Request/session-scoped replayable LLM/tool calls without adopting an engine | **sefia** |

## Sefia is a good fit when

- your workflow reads naturally as Python functions or service methods;
- selected calls should be LLM-backed but still typed like ordinary code;
- model/tool outputs must replay unchanged after a restart;
- a human may approve or provide input later;
- you do not want an `Agent` object, graph DSL, or workflow engine as the
  application model.

## Another tool may fit better when

- **Pydantic AI** — you want an explicit `Agent` abstraction with native provider
  tooling and ecosystem integrations.
- **Pydantic AI + Temporal / DBOS** — you want Pydantic AI's agent model plus a
  durable execution backend.
- **LangGraph** — the graph/state machine is the artifact you want to inspect and
  operate.
- **CrewAI** — the application is naturally a team of agents, roles, tasks, and
  flows.
- **Temporal / DBOS directly** — the problem is general durable execution: timers,
  workers, compensation, distributed fan-out, or auditable workflow history.

## Rule of thumb

If the flow is ordinary Python that has to pause and resume within a request/session
shape, sefia keeps it that way. If the primary artifact is an agent object, graph,
crew, or general workflow system, choose the layer that matches that boundary. The
reasoning is in [tradeoffs.md](./tradeoffs.md).
