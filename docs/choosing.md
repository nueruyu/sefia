# Choosing a stack: when to use sefia, and when not to

A decision guide. It names real tools and, where they fit your problem better, sends
you to them. sefia covers a deliberately narrow shape: the durable, human-in-the-loop,
typed-function agent turn on stateless HTTP. Most of this page is about when that shape
is *not* what you have.

> Caveat: the landscape moves fast and these tools ship constantly. Treat the
> characterizations as category-level and current as of writing, not as fixed
> facts about any library's latest release. When in doubt, check the tool.

## Start here (quick triage)

1. **Do you need durability / resume-across-restart at all?**
   No → you don't need most of this page. Use a direct provider SDK or a typed
   agent library (e.g. Pydantic AI) and move on.
2. **Is the flow long-horizon (days–weeks), cross-service with compensation, or
   distributed across machines?**
   Yes → you want a real workflow engine. Use **Temporal** (or **DBOS** if you
   prefer a Postgres-backed library over a cluster). sefia is the wrong tool.
3. **Do you want the control flow itself to be an explicit, inspectable artifact
   with complex branching/routing?**
   Yes → use a graph framework (**LangGraph**).
4. **Is the work open-ended and model-driven (the model loops, picks tools), and
   you want native per-provider tool-calling and a big ecosystem?**
   Yes → use a typed agent runtime (**Pydantic AI**); add its durable wrapper if
   you need resume.
5. **Otherwise — a typed Python call graph that must pause for a human and resume
   on a fresh request, with no engine to operate?**
   That's **sefia**.

## The options

### Direct provider SDK (no framework)
- **Use it when:** one-shot or stateless calls, a simple tool loop you're happy to
  own, no resume requirement. Nothing beats it for minimalism.
- **Not when:** you find yourself hand-rolling structured output, a tool loop,
  retries, and especially resume — at which point you're writing a framework.

### Pydantic AI (typed agent runtime)
- **Use it when:** you want a mature, widely-used typed agent library with **native
  tool-calling** across many providers, a big ecosystem, and model-driven loops.
  Durability is a first-class wrapper (`TemporalAgent` / `DBOSAgent`) away.
- **Not when:** you want durability *without* adopting an engine, or you want to
  avoid per-provider tool-calling/structured-output differences surfacing in your
  code. Then look at sefia.
- **Note:** this is the closest neighbor to sefia on typed ergonomics and a
  well-regarded one. The real difference is *how you get durability* (engine vs.
  native) and *whether provider tool-calling leaks* — see
  [tradeoffs.md](./tradeoffs.md).

### LangGraph (graph framework)
- **Use it when:** the flow is a state machine — complex branching, cycles, explicit
  shared state you want to inspect — and you want the graph as an operable artifact,
  with a built-in checkpointer + interrupt for durable pause.
- **Not when:** your logic is just ordinary Python control flow. Then authoring it
  as nodes/edges/`State` is overhead you don't need; a plain typed function (sefia)
  reads better.

### DBOS (durable functions on Postgres)
- **Use it when:** you want general-purpose durable execution (not just agents),
  you're **already on Postgres**, and the LLM layer (tool loop, structured output)
  can live in your application code on top of durable `@workflow`/`@step`.
- **Not when:** you want an LLM-native agent out of the box, or you don't want
  Postgres as a hard dependency. DBOS is light as engines go, so the gap to sefia
  is mostly "no engine, no mandatory Postgres, LLM-native" rather than raw weight.

### Temporal (distributed workflow engine)
- **Use it when:** long-horizon waits (days–weeks on durable timers), cross-service
  sagas with compensation, distributed fan-out across machines, or large-scale
  exactly-once with a queryable audit history. Temporal is appropriate when those
  guarantees are central to the workload.
- **Not when:** your workload is a request-scoped agent turn. Then a cluster +
  workers is infrastructure you'd operate for guarantees you don't use.

### Multi-agent frameworks (CrewAI, AutoGen, …)
- **Use them when:** your problem is several collaborating/role-playing agents
  negotiating a task. That orchestration shape is their focus.
- **Not when:** you have one durable typed pipeline. sefia is not a multi-agent
  framework and doesn't try to be.

### sefia
- **Use it when:** a durable, typed Python call graph that must **pause for a human
  and resume across a process restart**, served on a **plain stateless HTTP
  handler** with a sqlite/file/Postgres store and **no engine, cluster, or graph
  DSL** to operate. Provider-portable by design.
- **Not when:** any of the rows above describe you better — long-horizon/distributed
  (Temporal/DBOS), flow-as-artifact (LangGraph), native-tool-calling-first or
  ecosystem-first (Pydantic AI), multi-agent (CrewAI/AutoGen), or no durability
  needed at all (direct SDK).

## Capability matrix

| | Durable resume | No engine to operate | Stateless-HTTP native | Flow as explicit artifact | Long-horizon / distributed | Native provider tool-calling |
| --- | --- | --- | --- | --- | --- | --- |
| **Direct SDK** | ✗ (DIY) | ✓ | ✓ | ✗ | ✗ | ✓ |
| **Pydantic AI** | via engine wrapper | ✗ (adopts engine) | depends on engine | ✗ | via Temporal | ✓ |
| **LangGraph** | ✓ (checkpointer) | ✗ (graph runtime) | partial | ✓ | partial | ✓ |
| **DBOS** | ✓ | ✗ (Postgres) | ✓ | ✗ | partial | n/a (you build it) |
| **Temporal** | ✓ | ✗ (cluster+workers) | ✗ | ✓ | ✓ | n/a (you build it) |
| **sefia** | ✓ (native) | ✓ | ✓ | ✗ | ✗ | ✗ (unified schema) |

"✓/✗" are about the *default* shape, not what's achievable with effort. Two cells
are tradeoffs, not wins: sefia's "no native tool-calling" is a
deliberate portability bet with a real cost (no native parallel tools, prompt
caching to design for), and its "no long-horizon/distributed" is a scope choice,
not a gap to close.

## One-line answers

- **"I just want to call a model with types."** → Direct SDK or Pydantic AI.
- **"Mature, native tool-calling, big ecosystem."** → Pydantic AI.
- **"My flow is a real state machine I want to see."** → LangGraph.
- **"I'm on Postgres and want durable everything."** → DBOS.
- **"Days-long waits / cross-service saga / distributed."** → Temporal.
- **"Several agents collaborating."** → CrewAI / AutoGen.
- **"A durable, human-in-the-loop typed turn on a plain HTTP handler, no engine."**
  → sefia.

The reasoning behind sefia's narrow shape is in [tradeoffs.md](./tradeoffs.md).
