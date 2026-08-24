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
class WebToolkit:                                  # a plain class — no base, no decorator
    def __init__(self, http): self._http = http   # private = internal
    async def search(self, q: str) -> list[str]:   # public = tool, once granted
        """Search the web and return URLs."""
        ...

class ResearchService:                             # a plain class — no base, no decorator
    _web: Tools[WebToolkit]                        # the class-level annotation is the grant

    def __init__(self, web: WebToolkit):
        self._web = web

    @infer
    async def run(self, topic: str) -> Report: ...
```

- **A tool is a member of a field granted with the `Tools` alias.** No ambient
  authority: holding an object is not enough — the class-level field annotation
  must say `Tools[WebToolkit]`. No decorators, no registry, no strings, no base
  classes: `Tools[T]` is an `Annotated` alias, so checkers see plain `T` and every
  type stays an ordinary class or `Protocol`. Within a granted field, public =
  tool, private = internal (ordinary encapsulation).
- **Tool dependencies are expressed through classes.** Tools ride on the `@infer`
  method's receiver (`self`); every other parameter is task input. A grant is
  local to the holding site — the same class can be a toolkit in one service and
  inert data in another.
- **Narrow by type.** A concrete class exposes its public methods; a `Protocol`
  exposes only its declared members (`_web: Tools[ReadOnlyWeb]`). Annotate `self`
  with a plain surface `Protocol` to select one method's tools — the annotation
  itself is the opt-in, including for the instance's own private methods.
- **Discovery is static and fail-closed.** The surface is a pure function of
  declared types — an undeclared or unmarked field exposes nothing, and runtime
  values never widen it. A service's own methods are never tools unless a surface
  declares them; it becomes another agent's tool by being held in a granted field.
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
class UserInput:
    async def get(self, prompt: str) -> str:
        """Prompt the user; resume when input is available."""
        if provided := await self._pending.input_for(prompt):
            return provided
        await self._pending.record(prompt)
        raise InputRequired(prompt)                   # pause — durably

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

- **Uniform decisions remain the default.** The default uses a single schema
  (`final_answer | tool_calls`) for provider portability and full return-type
  expressiveness. Native tool calling is an explicit transport; the independently
  selected result transport may use structured output or a synthetic tool call. Full argument:
  [tradeoffs.md](./docs/tradeoffs.md).
- **Lighter than Temporal, not a replacement.** Single-process /
  resume-on-fresh-request, plus horizontal scale across independent sessions.
  Distributed single-workflow branches are out of scope.
- **Replay assumes determinism** between engraved steps — every replay engine's
  caveat.
- **Explicit capability gate.** Tools require the `Tools[...]` field annotation
  rather than "any public method of anything held". The cost is one alias per
  granted field; the benefit: a held member cannot leak as a tool by accident, and
  the surface is statically declared and type-checkable.
- **Pre-1.0.** The API will change.
