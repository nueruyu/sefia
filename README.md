# Sefia

**S**tateless **E**ngraved **F**unction **I**nference **A**bstraction

> LLM agents that pause for a human and resume after a restart, written as ordinary
> typed Python functions on a plain stateless HTTP handler — no workflow engine to
> run, just a store you already have (in-memory, a file, or your own database).

> The code in these docs shows the **release-target (1.0) API**. sefia is pre-1.0 and
> some surfaces still differ — see [Status](#status).

```python
from sefia import infer

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

| What you get | What you don't run or learn |
| --- | --- |
| LLM steps as plain typed functions (`@infer`) | an `Agent` object or a graph DSL |
| Tools = the public methods of the objects an agent holds | a tool registry or decorators |
| Runs that pause and resume across a restart (by replay) | a workflow engine, cluster, or worker |
| Human-in-the-loop over plain stateless HTTP | websockets or background daemons |
| One provider-portable output schema | per-provider native tool-calling quirks |

Under the hood, [glyff](https://github.com/nueruyu/glyff) content-addresses and replays
each call, so **pausing is just raising**. The store behind it is your choice:
in-memory, a file, or your own database.

For the reasoning behind these choices, see **[Design & Philosophy](./DESIGN.md)**
and **[Less to learn, less to leak, less to operate](./docs/why-less.md)**. For a
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
  [LiteLLM](https://github.com/BerriAI/litellm).

The replay engine underneath, [glyff](https://github.com/nueruyu/glyff), is installed
automatically.

## Quickstart

A typed agent that holds a tool, runs an inferred step, and persists its run.

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


class Researcher:
    def __init__(self, web: WebSearchTool):
        self._web = web                       # held dependency → its public methods are tools

    @infer
    async def run(self, topic: str) -> Report:
        """Research the topic with web search and produce a structured report."""
        ...


scope = SessionScope(session_dir=Path(".sessions"), model="gpt-4o")

async def main(topic: str) -> Report:
    agent = Researcher(web=WebSearchTool())
    async with scope.session(session_id="demo") as _:
        return await agent.run(topic)         # the engraved run can pause and resume
```

`SessionScope` wires the LLM client, the glyff session, and the store for you; drop to
`sefia.Session` directly when you want full control. For a step-by-step walk from here
to a human-in-the-loop agent that resumes over HTTP, see the
**[Quickstart tutorial](./docs/quickstart.md)**.

## Pause for a human, resume after a restart

A turn that pauses for a human and resumes after a restart, served on an ordinary
request/response handler: the pause is a tool that **raises**, and resume is calling
the endpoint again.

```python
from sefios.tools import HumanInputTool


class Assistant:
    def __init__(self, web: WebSearchTool, human: HumanInputTool):
        self._web = web
        self._human = human

    @infer
    async def run(self, task: str) -> Report:
        """Research the task, draft a report, ask the human to approve it, then finalize."""
        ...


agent = Assistant(web=WebSearchTool(), human=HumanInputTool())


@app.post("/sessions/{session_id}/turn")
async def turn(session_id: str, body: TurnBody):
    async with scope.session(session_id=session_id) as s:
        if body.answer is not None:
            await s.accept_input(body.answer)      # deliver the human's answer
        try:
            return {"status": "done", "report": await agent.run(body.task)}
        except NeedsInput as e:                     # the run paused; it will resume
            return {"status": "needs_input", "question": e.question}
```

When the human tool has no recorded answer it raises `NeedsInput`; the run pauses and
the handler returns "needs input". The answer arrives in a later request and is
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
| **Tools** | The public methods of the objects an agent holds. Public = tool, private = internal. Scoped to the holder; narrow with a `Protocol`. |
| **Pause & resume** | Every call is engraved (content-addressed) via glyff and replays on re-invocation; exceptions are non-terminal, so pausing = raising. |
| **Session** | The scope for a run. `SessionScope` (in `sefios`) is the configured front door; `sefia.Session` is the core primitive. |
| **Policies & middleware** | Observation (handlers, isolated) vs. control (middleware steers). The `sefios` defaults give a step cap and ready-made behaviors. |
| **Stores** | Where engraved progress and tool state live — memory, file, or your own backend. Your application database stays yours. |

A note on the design choice underneath all of this:
[statelessness as a tradeoff](./docs/notes/statelessness.md) (a neutral design note,
not a pitch).

## Documentation

Full index with a suggested reading path: **[docs/](./docs/)**.

- **[Quickstart](./docs/quickstart.md)** — from one inferred function to a resumable
  HITL agent over HTTP, step by step.
- **[Design & Philosophy](./DESIGN.md)** — the thesis and the model in full.
- **[How it works](./docs/how-it-works.md)** — the mechanism behind `@infer`, with
  source references: the loop, the unified schema, and content-addressed replay.
- **[Architecture map](./docs/architecture.md)** — package layout, dependency
  direction, and "if you want to change X, look at Y" (handy for AI-assisted dev).
- **[Contributing](./CONTRIBUTING.md)** — setup, commands, and the development
  workflow.
- **[Less to learn, less to leak, less to operate](./docs/why-less.md)** — the
  positioning argument: concept surface, provider leakage, operational weight.
- **[Choosing a stack](./docs/choosing.md)** — "when to use what, and when not to use
  sefia".
- **[Use cases](./docs/usecases/)** — human-in-the-loop and approval-gated workflows,
  hand-rolled and across LangGraph / Pydantic AI / sefia.
- **[FAQ](./docs/faq.md)** — answers to the common objections and "how does it
  actually work" questions.
- **[Statelessness — a design note](./docs/notes/statelessness.md)** — the
  vendor-neutral tradeoff this all rests on.
- **[Examples](./examples/)** — runnable end-to-end agents.

## Status

Pre-1.0 — the API is unstable and will change. Parts of the design, notably the tool
model, are being finalized; see [DESIGN.md](./DESIGN.md) and the issue tracker for
what is settled and what is in flight.
