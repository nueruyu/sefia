# Sefia

**S**tateless **E**ngraved **F**unction **I**nference **A**bstraction

> Durable, resumable LLM agents as ordinary typed Python functions — pause for a
> human and survive a restart over plain stateless HTTP, with no workflow engine,
> no graph DSL, and no database to operate but a small store.

```python
from sefia import infer

@infer
async def summarize(article: str) -> Summary:
    """Summarize the article for a technical audience; note key uncertainty."""
    ...
```

That's the whole idea: an **`@infer` function is an abstract method whose
implementer happens to be an LLM.** The signature is the input contract, the return
type is the validated output contract, the docstring is the instruction, and the
body is `...`. The call site stays ordinary Python.

```python
brief    = await clarify(request)
sources  = await research(brief)
report   = await write(brief, sources)     # plain await, plain control flow
```

## What makes it different

- **LLM steps are functions.** `@infer` on a typed async function or method. No
  `Agent` object, no registry, no graph.
- **Tools are the public surface of the objects an agent holds.** A held dependency's
  public methods are its tools; private (`_`) methods stay internal. Ordinary OOP
  encapsulation, scoped to the object — no decorators, no global tool registry.
- **Durable execution, no engine.** Backed by
  [glyff](https://github.com/nueruyu/glyff): each call is content-addressed and
  replayed on re-invocation. Any exception is non-terminal — completed work commits,
  the interrupted call stays resumable. **Pausing is just raising.**
- **Human-in-the-loop over stateless HTTP.** A paused run survives process death:
  re-invoke in a fresh request and completed steps replay while the pending one runs.
  No worker, no websocket, no cluster — just a request, a response, and a store.
- **State is explicit.** Inputs in, outputs out; mutable state lives in a store only
  when a tool needs it.

For the reasoning behind these choices, see **[Design & Philosophy](./DESIGN.md)**
and **[Less to learn, less to leak, less to operate](./docs/why-less.md)**. For an
honest "use X if you want Y, sefia if you want Z" guide, see
**[Choosing a stack](./docs/choosing.md)**.

## Install

```bash
pip install 'sefios[litellm]'
```

- **`sefia`** — the core: `@infer`, the tool model, sessions, durability.
- **`sefios`** — the official batteries: the `SessionScope` front door, ready-made
  policies/middleware, and tools (human input, web search). The `[litellm]` extra
  pulls in **`sefia_litellm`** for provider support via
  [LiteLLM](https://github.com/BerriAI/litellm).

Durability is provided by [glyff](https://github.com/nueruyu/glyff), installed
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
        return await agent.run(topic)         # the engraved run is durable & resumable
```

`SessionScope` wires the LLM client, the glyff durability session, and the store for
you; drop to `sefia.Session` directly when you want full control. For a step-by-step
walk from here to a durable HITL agent over HTTP, see the
**[Quickstart tutorial](./docs/quickstart.md)**.

## Durable human-in-the-loop

The defining use case: a turn that pauses for a human and resumes after a restart,
served on an ordinary request/response handler. The pause is a tool that **raises**;
resume is just calling the endpoint again.

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


@app.post("/sessions/{session_id}/turn")
async def turn(session_id: str, body: TurnBody):
    # the human tool returns a recorded answer if present, else records the question and raises
    human = HumanInputTool(get_answer=answer_for(session_id, body.answer))
    agent = Assistant(web=WebSearchTool(), human=human)
    async with scope.session(session_id=session_id) as _:
        try:
            report = await agent.run(body.task)
            return {"status": "done", "report": report}
        except NeedsInput as e:               # the run paused durably
            return {"status": "needs_input", "question": e.question}
```

When the human tool has no answer yet it records the question and raises; the run
pauses **durably** and the handler returns "needs input". The answer arrives in a
later request, the same endpoint re-invokes, every completed LLM/tool call **replays
its exact output** (the approved draft is byte-for-byte the same), and only the
pending step runs. No checkpoint code, no step keys, no idempotency plumbing, no 202
dance — see [use case 01](./docs/usecases/01-human-in-the-loop.md) for the same turn
hand-rolled, and what collapses.

## Core concepts

| Concept | What it is |
| --- | --- |
| **`@infer`** | An abstract async method implemented by an LLM. Signature = contract, docstring = instruction, return type = validated output. |
| **Tools** | The public methods of the objects an agent holds. Public = tool, private = internal. Scoped to the holder; narrow with a `Protocol`. |
| **Durability** | Every call is engraved (content-addressed) via glyff and replays on re-invocation; exceptions are non-terminal, so pausing = raising. |
| **Session** | The durable scope for a run. `SessionScope` (in `sefios`) is the configured front door; `sefia.Session` is the core primitive. |
| **Policies & middleware** | Observation (handlers, isolated) vs. control (middleware steers). The `sefios` defaults give a step cap and ready-made behaviors. |
| **Stores** | Where engraved progress and tool state live — memory, file, or your own backend. Your application database stays yours. |

A note on the design choice underneath all of this:
[statelessness in durable execution](./docs/notes/statelessness.md) (a neutral design
note, not a pitch).

## Why not native tool-calling?

Sefia asks the model for one unified result shape (`final_answer | tool_calls`) and
uses strict structured output where the provider supports it, instead of binding to
each provider's native tool-calling. The win is **provider-portability and full
return-type expressiveness** with no per-provider semantics leaking into your code;
the honest cost is no native parallel tool calls and prompt caching as something to
design for rather than get for free. See [DESIGN.md](./DESIGN.md#non-goals--honest-tradeoffs).

## Documentation

- **[Quickstart](./docs/quickstart.md)** — from one inferred function to a durable
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
- **[Choosing a stack](./docs/choosing.md)** — honest "when to use what, and when
  not to use sefia".
- **[Use cases](./docs/usecases/)** — durable HITL and approval-gated workflows,
  hand-rolled and across paradigms.
- **[FAQ](./docs/faq.md)** — honest answers to the common objections and "how does
  it actually work" questions.
- **[Statelessness — a design note](./docs/notes/statelessness.md)** — the
  vendor-neutral tradeoff this all rests on.
- **[Examples](./examples/)** — runnable end-to-end agents.

## Status

Pre-1.0 — the API is unstable and will change. Parts of the design, notably the tool
model, are being finalized; see [DESIGN.md](./DESIGN.md) and the issue tracker for
what is settled and what is in flight.
