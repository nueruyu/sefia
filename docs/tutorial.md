# Tutorial: a resumable human-in-the-loop agent

A progressive walk from a single inferred function to a human-in-the-loop agent served
over HTTP that resumes after a restart. About fifteen minutes. For the short version,
see the [Quickstart](./quickstart.md).

> This tutorial is written against the **release-target API**. sefia is pre-1.0 and
> parts (notably the tool model) are being finalized, so names may shift before
> 1.0; see [DESIGN.md](../DESIGN.md) for what is settled.

## Install

```bash
pip install 'sefios[litellm]'
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
from sefia import infer
from sefios import SessionScope


class Summary(BaseModel):
    key_points: list[str]
    uncertainty: str


@infer
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
client, the durability session, and a file store under `.sessions/` for you.

## 2. Give it a tool

Tools are the **public methods of the objects an agent holds** — no decorator, no
registry. Make the work a method on an agent, hold a dependency, and its public
methods become callable by the inferred step.

```python
from sefios.tools import WebSearchTool


class Report(BaseModel):
    topic: str
    summary: str
    sources: list[str]


class Researcher:
    def __init__(self, web: WebSearchTool):
        self._web = web                 # held dependency → its public methods are tools

    @infer
    async def run(self, topic: str) -> Report:
        """Research the topic with web search and produce a structured report."""
        ...


async def main() -> None:
    agent = Researcher(web=WebSearchTool())
    async with scope.session(session_id="quickstart"):
        report = await agent.run("durable execution for LLM agents")
        print(report.summary)
```

`self._web` is held, so `WebSearchTool`'s public `search` method is offered to the
model; the private `_web` *field* is just storage. The model decides when to call the
tool. To expose a narrower surface than a class's full public API, hold it behind a
`Protocol` — only the protocol's declared members are offered.

## 3. Make it pause for a human — and survive a restart

This is the part that is painful to hand-roll. Add a human-input tool. When it has no
answer it records the question and **raises**; the run pauses *durably*. Because the
session is engraved, you can resume in a **completely new process** and the completed
steps replay instead of re-running.

```python
# hitl.py
import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel
from sefia import infer
from sefia.exceptions import NeedsInput          # raised when the run pauses
from sefios import SessionScope
from sefios.tools import HumanInputTool, WebSearchTool


class Report(BaseModel):
    topic: str
    summary: str


class Assistant:
    def __init__(self, web: WebSearchTool, human: HumanInputTool):
        self._web = web
        self._human = human

    @infer
    async def run(self, task: str) -> Report:
        """Research the task, draft a report, ask the human to approve it, then finalize."""
        ...


scope = SessionScope(session_dir=Path(".sessions"), model="gpt-4o")


async def main() -> None:
    answer = sys.argv[1] if len(sys.argv) > 1 else None   # pass the answer on resume
    agent = Assistant(web=WebSearchTool(), human=HumanInputTool())
    async with scope.session(session_id="approval-demo") as s:
        if answer is not None:
            await s.accept_input(answer)                  # deliver the answer on resume
        try:
            report = await agent.run("the state of durable LLM agents")
            print("DONE:", report.summary)
        except NeedsInput as e:
            print("NEEDS INPUT:", e.question)


asyncio.run(main())
```

Run it once with no answer — it researches, drafts, then pauses:

```bash
python hitl.py
# NEEDS INPUT: Here's the draft: "...". Approve it?
```

Now run it **again** (a fresh process) with the answer. The clarify/search/draft
steps are not re-run — they **replay their exact stored outputs**, so the model is
approving the *same* draft — and only the finalize step executes:

```bash
python hitl.py "yes, approve"
# DONE: ...
```

That second invocation could be on another machine, after a deploy, or days later.
There was no checkpoint code, no step keys, no idempotency bookkeeping — just a tool
that raised and a session that replays.

## 4. Serve it over HTTP

The same agent behind a stateless request/response handler. A pause returns
"needs input"; the answer arrives in a later request to the same session id, and the
run resumes. Nothing runs in the background between the two requests.

```python
# server.py
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from sefia.exceptions import NeedsInput
from sefios import SessionScope
from sefios.tools import HumanInputTool, WebSearchTool

# (Assistant, Report from hitl.py)

app = FastAPI()
scope = SessionScope(session_dir=Path(".sessions"), model="gpt-4o")
agent = Assistant(web=WebSearchTool(), human=HumanInputTool())


class TurnBody(BaseModel):
    task: str
    answer: str | None = None


@app.post("/sessions/{session_id}/turn")
async def turn(session_id: str, body: TurnBody):
    async with scope.session(session_id=session_id) as s:
        if body.answer is not None:
            await s.accept_input(body.answer)     # deliver the human's answer
        try:
            return {"status": "done", "report": await agent.run(body.task)}
        except NeedsInput as e:
            return {"status": "needs_input", "question": e.question}
```

```bash
uvicorn server:app
```

```bash
# first request — pauses for approval
curl -X POST localhost:8000/sessions/abc/turn \
  -H 'content-type: application/json' \
  -d '{"task": "the state of durable LLM agents"}'
# {"status":"needs_input","question":"Here's the draft ... Approve it?"}

# restart the server here if you like — the paused run survives

# second request — resumes and finalizes
curl -X POST localhost:8000/sessions/abc/turn \
  -H 'content-type: application/json' \
  -d '{"task": "the state of durable LLM agents", "answer": "yes, approve"}'
# {"status":"done","report":{...}}
```

The handler is an ordinary stateless endpoint. The durable run lives in the store
under `.sessions/`, not in the process — so killing and restarting the server
between the two requests changes nothing.

## What just happened

- An **`@infer`** function is an LLM-implemented abstract method; you compose them
  with plain `await`.
- **Tools** are the public methods of held objects — ordinary OOP, scoped to the
  holder.
- **Durability** is native: every call is engraved and replays on re-invocation, so
  pausing is a tool raising and resuming is calling again.
- It runs on a **stateless handler with a store**: no engine, worker, or graph.

## Next steps

- Swap the file store for your own backend, or drop to `sefia.Session` for full
  control over the LLM client, policies, and middleware.
- Read [use case 01](./usecases/01-human-in-the-loop.md) to see this same turn
  hand-rolled, and exactly what the framework removed.
- Read [Design & Philosophy](../DESIGN.md) and the [FAQ](./faq.md) for the model and
  the tradeoffs.
- For long-horizon "resume in N days" flows, add an external scheduler that re-calls
  the endpoint — see the [timer note in the FAQ](./faq.md#what-about-long-running-waits--timers).
