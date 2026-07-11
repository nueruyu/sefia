# Design

> Status: pre-1.0, API unstable. The code here shows the **release-target (1.0) API** —
> the design we are building toward; some surfaces still differ today (see the issue
> tracker).

**`@infer` turns a typed Python function into an LLM-backed call whose completed steps
are engraved and replay on re-invocation.** A paused run is just an engraved call that
hasn't finished, which is what makes durable human-in-the-loop over a plain stateless
HTTP handler possible — with no workflow engine and no graph DSL.

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

Sefia does not introduce an `Agent` object. An application may have services,
workflows, handlers, or plain functions. Sefia only changes what happens when an
`@infer` function is called: the function body is not executed; its signature,
return type, and docstring become the contract for an LLM-backed, replayable call.

```python
class WebToolkit:
    def __init__(self, http): self._http = http   # private = internal
    async def search(self, q: str) -> list[str]:   # public = tool
        """Search the web and return URLs."""
        ...

class ResearchService:
    def __init__(self, web: WebToolkit):
        self._web = web                            # held dependency = tool
    @infer
    async def run(self, topic: str) -> Report: ...
```

- **Tools = the public surface of held dependency objects.** Public = tool,
  private = internal (ordinary encapsulation). No decorators, no registry. Tools
  are scoped to the object that holds them.
- **Narrow by type.** A concrete class exposes its public methods; a `Protocol`
  exposes only its declared members.
- **A class's own methods aren't its own tools.** Its `@infer` methods are not
  offered back to itself, so a run can't recurse into itself. (A service object can
  still be held by another service and act as a dependency.)
- **A held field is a tool; an `@infer` argument is task input.** So held fields
  should be dependency objects, not unrelated state.
- **Batched calls run serially; overlap is opt-in.** When one model decision
  contains several tool calls they execute one after another, unless a tool
  method is marked `@concurrent` — the author's declaration that overlapping its
  calls with batch siblings is safe. Results are always awaited and recorded in
  request order; the marker never changes exposure, only scheduling.

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
        return await service.run(body.task)        # resumes where it paused
```

A paused run survives process death: re-invoke and the completed engraved work
replays while the pending step re-runs. No engine, graph, or websocket.

## How it relates

sefia keeps durability built into ordinary Python, rather than a graph to author
(LangGraph), an adopted engine (Pydantic AI's `TemporalAgent`/`DBOSAgent`, or DBOS and
Temporal directly), or a distributed cluster. It targets the lighter, single-flow,
request-scoped layer before you need a distributed workflow engine. The per-tool
comparison and "when to use which" are in [docs/choosing.md](./docs/choosing.md); the
tradeoffs behind the design are in [docs/tradeoffs.md](./docs/tradeoffs.md).

## Non-goals & tradeoffs

- **Not native tool-calling.** A single unified schema (`final_answer |
  tool_calls`) plus strict structured output where supported → provider-portable,
  full return-type expressiveness; at the cost of native parallel tools and some
  frontier-model tuning on complex agents. Concurrency and prompt caching are tracked
  on the issue tracker, not guaranteed. Full argument:
  [tradeoffs.md](./docs/tradeoffs.md).
- **Lighter than Temporal, not a replacement.** Single-process /
  resume-on-fresh-request, plus horizontal scale across independent sessions.
  Distributed single-workflow branches are out of scope.
- **Replay assumes determinism** between engraved steps — every replay engine's
  caveat.
- **Minimal core by convention** — the rule that held objects are dependencies whose
  public methods are tools (nothing unrelated) is not enforced today (may gain a
  static check).
- **Pre-1.0.** The API will change.
