# Sefia

**S**tateless **E**ngraved **F**unction **I**nference **A**bstraction

> sefia turns typed Python functions into durable, replayable LLM-backed calls.
> Because model and tool steps replay on re-invocation, a call can pause, resume after
> a restart, and drive human-in-the-loop flows over ordinary request/response handlers
> without a workflow engine or graph DSL.

```python
from pydantic import BaseModel
from sefios import domain


class Summary(BaseModel):
    key_points: list[str]
    uncertainty: str


infer = domain("myapp").infer


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
| Tools = public methods of `Tools[...]`-granted fields | a tool registry or decorators |
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
pip install 'sefios[litellm,sqlite]'
```

- **`sefia`** — the core: `@infer`, the tool model, sessions, and replay.
- **`sefios`** — the opinionated batteries: the `SessionScope` front door, ready-made
  policies/middleware, and tools (external input, web search). The `[litellm]` extra
  pulls in **`sefia_litellm`** for provider support via
  [LiteLLM](https://github.com/BerriAI/litellm). The `[cli]` and `[fastapi]` extras
  pull in **`sefia_typer`** / **`sefia_fastapi`** and unlock the `sefios.cli` /
  `sefios.fastapi` integrations — Typer and FastAPI apps with persisted sessions
  and human-in-the-loop pause/resume.

Live end-to-end compatibility is currently verified against OpenAI, Anthropic, and
Gemini. Other LiteLLM providers may work but are not yet covered by the live
compatibility suite.

The replay engine underneath, [glyff](https://github.com/nueruyu/glyff), is installed
automatically.

Persistence is process-local by default. The quickstart installs the `sqlite` extra
and selects `SQLitePersistence` explicitly so its sessions survive restarts.

**Import from `sefios`.** It re-exports the everyday authoring surface — the
`domain` / `concurrent` / `preview` / `policy` / `profile` decorators,
`Tools`, `AsRawText`, and `Policy` / `Profile` — alongside its own `SessionScope` and
batteries, so application code needs only `sefios`. Reach into `sefia` directly for the
extension seams (a custom policy, strategy, client, or tool collector) and tool-call
context helpers such as `current_tool_call_id_for`.

## Quickstart

A plain Python class that holds a dependency, runs an inferred step, and persists its
run.

```python
from pydantic import BaseModel
from sefios import SQLitePersistence, SessionScope, Tools, domain
from sefios.tools import WebSearch


class Report(BaseModel):
    topic: str
    summary: str
    sources: list[str]


infer = domain("myapp").infer

class ResearchService:
    _web: Tools[WebSearch]                # the field annotation grants the tools

    def __init__(self, web: WebSearch):
        self._web = web

    @infer
    async def run(self, topic: str) -> Report:
        """Research the topic with web search and produce a structured report."""
        ...


scope = SessionScope(
    model="gpt-4o",
    persistence=SQLitePersistence(),
)

async def main(topic: str) -> Report:
    service = ResearchService(web=WebSearch())
    async with scope.session(session_id="demo") as _:
        return await service.run(topic)       # the engraved run can pause and resume
```

`SessionScope` wires the LLM client, the glyff session, and a shared SQLite database
for durable execution and session state; drop to
`sefia.Session` directly when you want full control. The **[tutorial](./docs/tutorial.md)**
builds this into a human-in-the-loop service that resumes over HTTP.

## Pause for a human, resume after a restart

A turn that pauses for a human and resumes after a restart, served on an ordinary
request/response handler: the pause is a tool that **raises**, and resume is calling
the endpoint again.

This example uses the FastAPI integration with SQLite persistence, so install both
extras — `pip install 'sefios[litellm,fastapi,sqlite]'`.

```python
from sefios import SQLitePersistence, Tools, domain
from sefios.fastapi import SefiaHTTP
from sefios.fastapi.exceptions import InputRequired
from sefios.tools import Input, WebSearch


infer = domain("myapp").infer

class ResearchService:
    _web: Tools[WebSearch]
    _input: Tools[Input]

    def __init__(self, web: WebSearch, input_tool: Input):
        self._web = web
        self._input = input_tool

    @infer
    async def run(self, task: str) -> Report:
        """Research the task, draft a report, ask the human to approve it, then finalize."""
        ...


api = SefiaHTTP(
    model="gpt-4o",
    persistence=SQLitePersistence(),
)
research_service = ResearchService(web=WebSearch(), input_tool=api.input_tool)


@app.post("/sessions")
def create_session():
    return {"session_id": api.create_session()}


@app.post("/sessions/{session_id}/turn")
async def turn(session_id: str, body: TurnBody):
    try:
        async with api.session(session_id=session_id) as session:
            if body.input is not None:
                await session.accept_input(body.input)
            report = await research_service.run(body.task)
            return {"status": "done", "report": report}
    except InputRequired as e:
        return {"status": "needs_input", "prompt": e.prompt}
```

When the input tool has no recorded input it raises `InputRequired`; `SefiaHTTP`
publishes the pause as an SSE event and re-raises it after the session context exits,
and the handler returns "needs input". The input arrives in a later request and is delivered
with `session.accept_input`; the same endpoint re-invokes, every completed LLM/tool
call **replays its exact output** (the approved draft is byte-for-byte the same), and
only the pending step runs. You write no checkpoint code, step keys, idempotency
plumbing, or 202 dance; see
[use case 01](./docs/usecases/01-human-in-the-loop.md) for the same turn hand-rolled,
and what it removes.

## Core concepts

| Concept | What it is |
| --- | --- |
| **`@infer`** | An abstract async method implemented by an LLM. Signature = contract, docstring = instruction, return type = validated output. |
| **Tools** | Public methods of a field granted with the `Tools[...]` annotation (`_web: Tools[WebToolkit]`). The wrapped type stays a plain class; narrow by granting through a `Protocol`. No ambient authority; the grant is local to the holder. Batched calls run serially unless a method is marked `@concurrent`. |
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
