# Design & Philosophy

> Status: pre-1.0, API unstable. The code here shows the **release-target (1.0) API** —
> the design we are building toward; parts, notably the tool model, are still in
> progress and some surfaces differ today (see the issue tracker).

**LLM agents that pause for a human and resume after a restart, written as ordinary
typed Python functions — human-in-the-loop over plain stateless HTTP, without a
workflow engine or graph DSL.**

## Thesis

- **`@infer` = an abstract method implemented by an LLM.** Signature = input
  contract; return type = validated output contract; docstring = instruction;
  body is `...`. An ordinary "declared, not yet implemented" method, with the LLM
  as the implementer.
- **Resumable execution, no engine.** Backed by
  [glyff](https://github.com/nueruyu/glyff): engraved calls are content-addressed
  (call identity + args) and replayed on re-invocation. Any exception is
  non-terminal — completed work commits, the interrupted call stays resumable, the
  exception propagates. So **pausing is just raising.**
- **HITL = a tool that raises.** No input yet → record the question, raise → the
  run pauses durably → the handler returns "needs input" + a session id →
  re-invoke resumes. Stateless request/response; **survives process death.** No
  special runtime protocol, websocket, or workflow engine.

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
- **An agent's own methods aren't its own tools.** Its `@infer` methods are not
  offered back to itself, so a run can't recurse into itself. (The agent object can
  still be held by another agent and act as its tool.)
- **A held field is a tool; an `@infer` argument is task input.** So an agent's
  fields are all tool objects, and nothing unrelated.

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
| DBOS        | Durable functions (Postgres) | yes               | general durable execution; the LLM layer stays application code |
| Temporal    | Distributed workflows        | yes               | distributed workflow infra across services |
| **sefia**   | **Typed async functions**    | **yes**           | **durable LLM steps as plain Python, lightweight** |

The short version: LangGraph and sefia both keep durability built in (a graph to
author vs. plain Python); Pydantic AI reaches durable HITL via native deferred tools or
an adopted engine; DBOS and Temporal are general engines you run underneath. The
per-tool detail, the tradeoffs, and "when to use which" are in
[docs/choosing.md](./docs/choosing.md) and [docs/tradeoffs.md](./docs/tradeoffs.md).

## Non-goals & tradeoffs

- **Not native tool-calling.** A single unified schema (`final_answer |
  tool_calls`) plus strict structured output where supported → provider-portable,
  full return-type expressiveness; at the cost of native parallel tools and some
  frontier-model tuning on complex agents. Concurrency and prompt caching are tracked
  on the issue tracker, not guaranteed. Full argument:
  [tradeoffs — provider leakage](./docs/tradeoffs.md#2-provider-leakage).
- **Lighter than Temporal, not a replacement.** Single-process /
  resume-on-fresh-request, plus horizontal scale across independent sessions.
  Distributed single-workflow branches are out of scope.
- **Replay assumes determinism** between engraved steps — every replay engine's
  caveat.
- **Minimal core by convention** — the rule that an agent's held objects are all
  tools (nothing unrelated) is not enforced today (may gain a static check).
- **Pre-1.0.** The API will change.
