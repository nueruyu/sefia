# Design & Philosophy

> Status: pre-1.0, API unstable. The code here shows the **release-target (1.0) API** —
> the design we are building toward; parts, notably the tool model, are still in
> progress and some surfaces differ today (see the issue tracker).

**Durable, resumable LLM agents as ordinary typed Python functions — HITL over
plain stateless HTTP, no workflow engine, no graph DSL.**

## Thesis

- **`@infer` = an abstract method implemented by an LLM.** Signature = input
  contract; return type = validated output contract; docstring = instruction;
  body is `...`. An ordinary "declared, not yet implemented" method, with the LLM
  as the implementer.
- **Durable execution, no engine.** Backed by
  [glyff](https://github.com/nueruyu/glyff): engraved calls are content-addressed
  (call identity + args) and replayed on re-invocation. Any exception is
  non-terminal — completed work commits, the interrupted call stays resumable, the
  exception propagates. So **pausing is just raising.**
- **HITL = a tool that raises.** No input yet → record the question, raise → the
  run pauses durably → the handler returns "needs input" + a session id →
  re-invoke resumes. Stateless request/response; **survives process death.** No
  special exception type, websocket, or engine.

## The model: only standard Python vocabulary

```python
class WebToolkit:
    def __init__(self, http): self._http = http   # private = internal
    async def search(self, q: str) -> list[str]:   # public = tool
        """Search the web and return URLs."""
        ...

class ResearchAgent:
    def __init__(self, web: WebToolkit):
        self._web = web                            # held dependency = tool
    @infer
    async def run(self, topic: str) -> Report: ...
```

- **Tools = the public surface of the objects an agent holds.** Public = tool,
  private = internal (ordinary encapsulation). No decorators, no registry. Tools
  are scoped to the object that holds them.
- **Narrow by type.** A concrete class exposes its public methods; a `Protocol`
  exposes only its declared members.
- **An agent consumes tools, never provides them.** No self-tools → no
  self-recursion.
- **A held field is a tool; an `@infer` argument is task input.** An agent holds
  only its tools.

## Principles

- **Observation ≠ control.** Handlers observe (they cannot steer; their exceptions
  are isolated); middleware steers. Two seams.
- **State is explicit.** In = arguments, out = return value; mutable state lives in
  stores only when a tool needs it.
- **Recoverable failures resume, not fail.** A transient provider error or an
  invalid response is a non-terminal exception → re-invoke re-runs only that step;
  completed steps replay.
- **The core is unopinionated; batteries live in `sefios`** (a default step cap,
  ready-made policies, the `SessionScope` front door, the HTTP integration). Drop
  to the core anytime.

## Durability & resumable HITL

```python
class HumanInput:
    async def ask(self, question: str) -> str:
        """Ask the user; resume when an answer is available."""
        if answer := await self._pending.answer_for(question):
            return answer
        await self._pending.record(question)
        raise NeedsInput(question)                 # pause — durably

@app.post("/sessions/{id}/turn")
async def turn(id, body):
    async with scope.session(session_id=id) as s:
        await s.accept_input(body.input)
        return await agent.run(body.task)          # resumes where it paused
```

A paused run survives process death: re-invoke and the completed engraved work
replays while the pending step re-runs. No engine, graph, or websocket.

## How it compares

| Tool        | Shape                        | Durable           | Best fit |
| ----------- | ---------------------------- | ----------------- | -------- |
| LangGraph   | Graph (nodes/edges/state)    | yes               | explicit state machines, complex routing |
| Pydantic AI | Agent objects                | native or via Temporal/DBOS | typed agent apps on a runtime |
| DBOS        | Durable functions (Postgres) | yes               | general durable execution; you build the LLM layer |
| Temporal    | Distributed workflows        | yes               | distributed workflow infra across services |
| **sefia**   | **Typed async functions**    | **yes**           | **durable LLM steps as plain Python, lightweight** |

- **vs LangGraph** — no graph to author; agent logic stays a normal Python call
  graph.
- **vs Pydantic AI** — closest on typed ergonomics, and a well-regarded one. Its
  durable HITL is either native *deferred tools* (the run returns, you persist the
  message history, a new run resumes with the result) or an adopted engine
  (first-class `TemporalAgent` / `DBOSAgent` wrappers). sefia's durability is native
  to the `@infer` model (pausing is just raising, resuming is re-invoking) on a
  plain stateless handler with a sqlite/file store, no message-history threading and
  no engine.
- **vs DBOS** — also code-first, non-graph, durable. The difference is altitude:
  DBOS gives durable *functions* (you build the LLM loop, tools, structured output
  yourself); sefia gives a durable *agent* with those built in, and needs no
  Postgres. The moat vs DBOS is the LLM-native layer, not storage weight (DBOS is
  itself light).
- **vs Temporal** — Temporal is a distributed workflow engine (cluster + workers);
  sefia/glyff cover the lighter, single-flow, request-scoped part before that.

For the positioning argument in full (concept surface, provider leakage, and
operational weight) see [Less to learn, less to leak, less to operate](./docs/why-less.md).
For a "use this if you want X, use sefia if you want Y" decision guide, see
[Choosing a stack](./docs/choosing.md).

## Non-goals & tradeoffs

- **Not native tool-calling.** A single unified schema (`final_answer |
  tool_calls`) plus strict structured output where supported → provider-portable,
  full return-type expressiveness; at the cost of native parallel tools and some
  frontier-model tuning on complex agents. Concurrency and prompt caching are tracked
  on the issue tracker, not guaranteed. Full argument:
  [why-less — less to leak](./docs/why-less.md#2-less-to-leak--provider-concerns-staying-out-of-your-abstraction).
- **Lighter than Temporal, not a replacement.** Single-process /
  resume-on-fresh-request, plus horizontal scale across independent sessions.
  Distributed single-workflow branches are out of scope.
- **Replay assumes determinism** between engraved steps — every replay engine's
  caveat.
- **Minimal core by convention** — "an agent holds only its tools" is not enforced
  today (may gain a static check).
- **Pre-1.0.** The API will change.
