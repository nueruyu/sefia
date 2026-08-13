# Tutorial: a resumable human-in-the-loop service

A progressive walk from a single inferred function to a human-in-the-loop service
served over HTTP that resumes after a restart. About fifteen minutes. For the minimal
example, see the [README](../README.md).

> This tutorial is written against the **release-target API**. sefia is pre-1.0 and
> parts (notably the tool model) are being finalized, so names may shift before
> 1.0; see [DESIGN.md](../DESIGN.md) for what is settled.

## Install

The tutorial builds up to the CLI and HTTP integrations, so install their extras
alongside the provider:

```bash
pip install 'sefios[litellm,cli,fastapi]'
```

Set whatever credentials your model needs (LiteLLM reads provider env vars):

```bash
export OPENAI_API_KEY=sk-...
```

## 1. Your first inferred function

An `@infer` function is an abstract method whose implementer is an LLM: the
signature is the input/output contract, the docstring is the instruction, the body
is `...`. You run it inside a **session**, which gives it durability and a store.

```python
# quickstart.py
import asyncio
from pathlib import Path

from pydantic import BaseModel
from sefios import SessionScope, domain


class Summary(BaseModel):
    key_points: list[str]
    uncertainty: str


infer = domain("com.example.quickstart", version="1").infer

@infer(name="summarize")
async def summarize(article: str) -> Summary:
    """Summarize the article for a technical audience; note key uncertainty."""
    ...


scope = SessionScope(session_dir=Path(".sessions"), model="gpt-4o")


async def main() -> None:
    async with scope.session(session_id="quickstart"):
        result = await summarize("Large language models ...")
        print(result.key_points)


asyncio.run(main())
```

```bash
python quickstart.py
```

The body never runs. sefia sends the signature, docstring, and arguments to the
model, then validates the response into a `Summary`. `SessionScope` wired the LLM
client, the durability session, and a file store under `.sessions/` for you. For the
full rules on arguments, service members, tools, and return types, see
[The `@infer` contract](./infer-contract.md).

## 2. Give it a tool

Tools are the **public methods of fields granted with the `Tools[...]` annotation**
— no decorator, no registry, no base class. Hold a dependency in a class-level field
annotated `Tools[...]`, and its public methods become callable by the inferred step.

```python
from sefios import Tools
from sefios.tools import WebSearch


class Report(BaseModel):
    topic: str
    summary: str
    sources: list[str]


class ResearchService:
    _web: Tools[WebSearch]      # the field annotation is the grant

    def __init__(self, web: WebSearch):
        self._web = web

    @infer(name="ResearchService.run")
    async def run(self, topic: str) -> Report:
        """Research the topic with web search and produce a structured report."""
        ...


async def main() -> None:
    service = ResearchService(web=WebSearch())
    async with scope.session(session_id="quickstart"):
        report = await service.run("durable execution for LLM applications")
        print(report.summary)
```

`_web` is granted, so `WebSearch`'s public `search` method is offered to the
model, which decides when to call it. Checkers treat `Tools[WebSearch]` as plain
`WebSearch`, and `WebSearch` itself is an ordinary class. A held member
without the grant — a config, a store — is never exposed, so there is no ambient
authority. To expose a narrower surface than a class's full public API, grant
through a `Protocol` (`_web: Tools[ReadOnlyWeb]`): only the protocol's declared
members are offered.

When the model requests several tool calls in one step, they run one at a time. A
tool that is safe to overlap with the other calls in its batch — a pure read like a
search — can be marked with `@concurrent` (`from sefios import concurrent`) on the
method; consecutive marked calls then run concurrently, and their results still come
back in request order. Leave tools unmarked when their side-effect ordering matters.

### Tool scope is the service boundary

A service class can have more than one `@infer` method. That is useful when the
methods share the same domain and the same narrow tool surface.

But tools are collected from the bound instance and the dependency objects it holds,
so every `@infer` method on the service should be allowed to see that tool surface.
If one operation needs broader, write-capable, or unrelated tools, split it into
another service — or annotate that one method's `self` with a plain surface
`Protocol` to select just its tools.

A good rule of thumb: if you want to tell one `@infer` method "do not use this tool",
narrow its `self`, or move that tool to a different service.

## 3. Make it pause for a human - and survive a restart

This is the part that is painful to hand-roll. Add an input tool through the CLI
facade. When it has no input it records the prompt and **raises**; `SefiaCLI`
renders the prompt and exits cleanly. Because the session is engraved, you can resume
in a **completely new process** and the completed steps replay instead of re-running.

```python
# hitl_cli.py
import asyncio
from pathlib import Path

import typer
from pydantic import BaseModel
from sefios import Tools, domain
from sefios.cli import SefiaCLI
from sefios.tools import Input, WebSearch


class Report(BaseModel):
    topic: str
    summary: str


infer = domain("com.example.research", version="1").infer

class ResearchService:
    _web: Tools[WebSearch]
    _input: Tools[Input]

    def __init__(self, web: WebSearch, input_tool: Input):
        self._web = web
        self._input = input_tool

    @infer(name="ResearchService.run")
    async def run(self, task: str) -> Report:
        """Research the task, draft a report, ask the human to approve it, then finalize."""
        ...


app = typer.Typer()
cli = SefiaCLI(session_dir=Path(".sessions"), model="gpt-4o")
service = ResearchService(web=WebSearch(), input_tool=cli.input_tool)


@app.command()
def run(answer: str | None = None) -> None:
    async def _run() -> None:
        async with cli.session(session_id="approval-demo") as session:
            await session.accept_input(answer)
            report = await service.run("the state of durable LLM applications")
            print("DONE:", report.summary)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
```

Run it once with no answer; it researches, drafts, then pauses:

```bash
python hitl_cli.py
# [INPUT_REQUIRED:<interaction_id>] Here's the draft: "...". Approve it?
```

Now run it **again** (a fresh process) with the answer. The clarify/search/draft
steps are not re-run; they **replay their exact stored outputs**, so the model is
approving the *same* draft, and only the finalize step executes:

```bash
python hitl_cli.py "yes, approve"
# DONE: ...
```

That second invocation could be on another machine, after a deploy, or days later.
There was no checkpoint code, no step keys, no idempotency bookkeeping; just a tool
that raised and a session that replays.

## 4. Serve it over HTTP

The same service behind a stateless request/response handler. A pause returns
"needs input"; the input arrives in a later request to the same session id, and the
run resumes. Nothing runs in the background between the two requests.

```python
# server.py
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from sefios.fastapi import SefiaHTTP
from sefios.fastapi.exceptions import InputRequired
from sefios.tools import WebSearch

# (ResearchService, Report from hitl_cli.py)

app = FastAPI()
api = SefiaHTTP(session_dir=Path(".sessions"), model="gpt-4o")
research_service = ResearchService(web=WebSearch(), input_tool=api.input_tool)


class TurnBody(BaseModel):
    task: str
    input: str | None = None


@app.post("/sessions")
def create_session():
    return {"session_id": api.create_session()}


@app.post("/sessions/{session_id}/turn")
async def turn(session_id: str, body: TurnBody):
    try:
        async with api.session(session_id=session_id) as session:
            await session.accept_input(body.input)
            report = await research_service.run(body.task)
            return {"status": "done", "report": report}
    except InputRequired as e:
        return {"status": "needs_input", "prompt": e.prompt}
```

```bash
uvicorn server:app
```

```bash
# create a session
SID=$(curl -s -X POST localhost:8000/sessions | python -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

# first request: pauses for approval
curl -X POST localhost:8000/sessions/$SID/turn \
  -H 'content-type: application/json' \
  -d '{"task": "the state of durable LLM applications"}'
# {"status":"needs_input","prompt":"Here's the draft ... Approve it?"}

# restart the server here if you like; the paused run survives

# second request: resumes and finalizes
curl -X POST localhost:8000/sessions/$SID/turn \
  -H 'content-type: application/json' \
  -d '{"task": "the state of durable LLM applications", "input": "yes, approve"}'
# {"status":"done","report":{...}}
```

The handler is an ordinary stateless endpoint. The durable run lives in the store
under `.sessions/`, not in the process, so killing and restarting the server
between the two requests changes nothing.

## What just happened

- An **`@infer`** function is an LLM-implemented abstract method; you compose them
  with plain `await`.
- **Tools** are the public methods of `Tools[...]`-granted fields — ordinary OOP
  plus one annotation, scoped to the holder, no ambient authority.
- **Durability** is native: every call is engraved and replays on re-invocation, so
  pausing is a tool raising and resuming is calling again.
- It runs on a **stateless handler with a store**: no engine, worker, or graph.

## Next steps

- Swap the file store for your own backend, or drop to `sefia.Session` for full
  control over the LLM client, policies, and middleware.
- Read [The `@infer` contract](./infer-contract.md) for the rules on arguments,
  service members, tool methods, and return types.
- Read [use case 01](./usecases/01-human-in-the-loop.md) to see this same turn
  hand-rolled, and exactly what the framework removed.
- Read [Design](../DESIGN.md) and the [FAQ](./faq.md) for the model and
  the tradeoffs.
- For long-horizon "resume in N days" flows, add an external scheduler that re-calls
  the endpoint — see the [timer note in the FAQ](./faq.md#what-about-long-running-waits--timers).
