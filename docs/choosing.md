# Choosing a stack

sefia is narrow in its abstraction — typed Python calls made durable and replayable —
not in what you can build with it. Where another tool fits better, use it.

> Category-level and current as of writing; these tools move fast, so check each one.

| Need | Consider |
| --- | --- |
| One-shot typed model calls | a provider SDK, or Pydantic AI |
| Model-driven agent runtime with native tool-calling | Pydantic AI |
| An explicit graph / state machine as the artifact | LangGraph |
| General durable execution on Postgres | DBOS |
| Distributed workflows, long timers, cross-service compensation | Temporal |
| Durable typed Python calls that pause/resume over request/response | **sefia** |

## sefia is a good fit when

- your workflow reads naturally as a Python call graph;
- model/tool outputs must replay unchanged after a restart;
- a human may approve or supply input later;
- you don't want to operate a workflow engine for this flow.

## sefia is not the fit when

- the graph itself is the artifact you want to operate → LangGraph;
- the flow spans services and needs compensation, or long self-firing timers, or
  distributed fan-out → Temporal / DBOS;
- provider-native tool-calling (parallel calls, per-provider tuning) matters more than
  portability → Pydantic AI.

The reasoning behind these tradeoffs is in [tradeoffs.md](./tradeoffs.md).
