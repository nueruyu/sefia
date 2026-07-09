# Sefia

**S**tateless **E**ngraved **F**unction **I**nference **A**bstraction

> sefia turns typed Python functions into durable, replayable LLM-backed calls.
> Because model and tool steps replay on re-invocation, a call can pause, resume after
> a restart, and drive human-in-the-loop flows over ordinary request/response handlers
> — with no workflow engine or graph DSL.

```python
from pydantic import BaseModel
from sefia import infer


class Summary(BaseModel):
    key_points: list[str]
    uncertainty: str


@infer
async def summarize(article: str) -> Summary:
    """Summarize the article for a technical audience; note key uncertainty."""
    ...
```

An **`@infer` function is an abstract method whose implementer is an LLM.** The
signature is the input contract, the return type is the validated output contract,
the docstring is the instruction, and the body is `...`. The call site stays ordinary
Python.

```python
brief    = await clarify(request)
sources  = await research(brief)
report   = await write(brief, sources)     # plain await, plain control flow
```

## What makes it different

You compose durable LLM steps with ordinary Python: typed functions, plain `await`,
replayable model/tool execution underneath.

| What you get | What you don't run or learn |
| --- | --- |
| LLM steps as plain typed functions (`@infer`) | an `Agent` object or a graph DSL |
| Tools = the public methods of held dependencies | a tool registry or decorators |
| Runs that pause and resume across a restart (by replay) | a workflow engine, cluster, or worker |
| Human-in-the-loop over plain stateless HTTP | websockets or background daemons |
| One provider-portable output schema | per-provider native tool-calling quirks |

Under the hood, [glyff](https://github.com/nueruyu/glyff) content-addresses and replays
each call, so **pausing is just raising**. The store behind it is your choice:
in-memory, a file, or your own database.

For the reasoning behind these choices, see **[Design](./DESIGN.md)**
and **[the positioning argument](./docs/tradeoffs.md)**. For a
"use X if you want Y, sefia if you want Z" guide, see
**[Choosing a stack](./docs/choosing.md)**.

## Install

```bash
pip install 'sefios[litellm]'
```

- **`sefia`** — the core: `@infer`, the tool model, sessions, and replay.
- **`sefios`** — the official batteries: the `SessionScope` front door, ready-made
  policies/middleware, and tools (human input, web search). The `[litellm]` extra
  pulls in **`sefia_litellm`** for provider support via
  [LiteLLM](https://github.com/BerriAI/litellm). The `[cli]` and `[fastapi]` extras
  pull in **`sefia_typer`** / **`sefia_fastapi`** and unlock the `sefios.cli` /
  `sefios.fastapi` integrations — Typer and FastAPI apps with persisted sessions
  and human-in-the-loop pause/resume.

The replay engine underneath, [glyff](https://github.com/nueruyu/glyff), is installed
automatically.

## Quickstart

A plain Python class that holds a dependency, runs an inferred step, and persists its
run.

```python
from pathlib import Path
from pydantic import BaseModel
from sefia import infer
from sefios import SessionScope
from sefios.tools import WebSearchTool


class Report(BaseModel):
    topic: str
    summary: str
    sources: list[str]


class ResearchService:
    def __init__(self, web: WebSearchTool):
        self._web = web                       # held dependency → its public methods are tools

    @infer
    async def run(self, topic: str) -> Report:
        """Research the topic with web search and produce a structured report."""
        ...


scope = SessionScope(session_dir=Path(".sessions"), model="gpt-4o")

async def main(topic: str) -> Report:
    service = ResearchService(web=WebSearchTool())
    async with scope.session(session_id="demo") as _:
        return await service.run(topic)       # the engraved run can pause and resume
```

`SessionScope` wires the LLM client, the glyff session, and the store for you; drop to
`sefia.Session` directly when you want full control. The **[tutorial](./docs/tutorial.md)**
builds this into a human-in-the-loop service that resumes over HTTP.

## Pause for a human, resume after a restart

A turn that pauses for a human and resumes after a restart, served on an ordinary
request/response handler: the pause is a tool that **raises**, and resume is calling
the endpoint again.

```python
from sefios.tools import InputTool


class ResearchService:
    def __init__(self, web: WebSearchTool, input_tool: InputTool):
        self._web = web
        self._input = input_tool

    @infer
    async def run(self, task: str) -> Report:
        """Research the task, draft a report, ask the human to approve it, then finalize."""
        ...


research_service = ResearchService(web=WebSearchTool(), input_tool=InputTool())


@app.post("/sessions/{session_id}/turn")
async def turn(session_id: str, body: TurnBody):
    async with scope.session(session_id=session_id) as s:
        if body.input is not None:
            await s.accept_input(body.input)       # deliver the user's input
        try:
            return {"status": "done", "report": await research_service.run(body.task)}
        except NeedsInput as e:                     # the run paused; it will resume
            return {"status": "needs_input", "prompt": e.prompt}
```

When the input tool has no recorded input it raises `NeedsInput`; the run pauses and
the handler returns "needs input". The input arrives in a later request and is
delivered with `accept_input`; the same endpoint re-invokes, every completed LLM/tool
call **replays its exact output** (the approved draft is byte-for-byte the same), and
only the pending step runs. You write no checkpoint code, step keys, idempotency
plumbing, or 202 dance — see
[use case 01](./docs/usecases/01-human-in-the-loop.md) for the same turn hand-rolled,
and what it removes.

## Core concepts

| Concept | What it is |
| --- | --- |
| **`@infer`** | An abstract async method implemented by an LLM. Signature = contract, docstring = instruction, return type = validated output. |
| **Tools** | The public methods of held dependency objects. Public = tool, private = internal. Scoped to the holder; narrow with a `Protocol`. |
| **Pause & resume** | Every call is engraved (content-addressed) via glyff and replays on re-invocation; exceptions are non-terminal, so pausing = raising. |
| **Session** | The scope for a run. `SessionScope` (in `sefios`) is the configured front door; `sefia.Session` is the core primitive. |
| **Policies & middleware** | Observation (handlers, isolated) vs. control (middleware steers). The `sefios` defaults give a step cap and ready-made behaviors. |
| **Stores** | Where engraved progress and tool state live — memory, file, or your own backend. Your application database stays yours. |

## Documentation

- **[Tutorial](./docs/tutorial.md)** — build a resumable human-in-the-loop service, step
  by step.
- **[Design](./DESIGN.md)** — the thesis and non-goals.
- **[How it works](./docs/how-it-works.md)** — the runtime mechanism.
- **[Docs index](./docs/)** — everything else (comparisons, FAQ, use cases,
  architecture), with a suggested reading path.

## Status

Pre-1.0 — the API is unstable and will change. The code in these docs targets the 1.0
API; some surfaces still differ today. See [DESIGN.md](./DESIGN.md) and the issue
tracker for what is settled and what is in flight.
