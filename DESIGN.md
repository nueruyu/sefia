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
class WebToolkit(Tools):                           # `Tools` = callable by the model
    def __init__(self, http): self._http = http   # private = internal
    async def search(self, q: str) -> list[str]:   # public + Tools = a tool
        """Search the web and return URLs."""
        ...

@dataclass
class ResearchService:
    _web: WebToolkit                               # held dependency, gated by Tools
    @infer
    async def run(self, topic: str) -> Report: ...
```

- **A tool is a member of a `Tools`-bearing type, reached from a capability
  parameter.** No ambient authority: holding an object is not enough — its declared
  type must carry the `Tools` role (by inheriting `Tools`, or `Annotated[T, Tools]`
  at the field for a type you can't edit). No decorators, no registry, no strings —
  one base class. Within a gated type, public = tool, private = internal (ordinary
  encapsulation).
- **Capability parameters carry tools; arguments are task input.** `self`/`cls`
  carry the held-dependency surface by convention; any other parameter carries tools
  only if its declared type bears the role — so a plain function gets tools too:
  `async def run(kit: WebToolkit, topic: str)`.
- **Narrow by type.** A concrete class exposes its public methods; a `Protocol`
  exposes only its declared members. Annotate `self` with a role-bearing surface
  protocol to shape or restrict one method's tools (and to opt a private method in).
- **Discovery is a pure function of static declarations** — fail-closed. A field
  with no `Tools`-bearing declared type exposes nothing; runtime values never widen
  the surface. A plain service class does not bear `Tools`, so its own methods
  (including its `@infer` methods) are never offered back to itself; it becomes
  another agent's tool only by declaring `Tools` and being held as a dependency.

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
        raise NeedsInput(prompt)                   # pause — durably

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
- **Explicit capability gate.** Tools are gated by the `Tools` role marker rather
  than "any public method of anything held" — no ambient authority. The cost is one
  base class per toolkit (or an `Annotated[T, Tools]` at a field); the benefit is
  that a held member cannot leak as a tool by accident, and the surface is
  statically declared and type-checkable.
- **Pre-1.0.** The API will change.
