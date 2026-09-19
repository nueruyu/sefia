# Tutorial: a resumable human-in-the-loop service

A progressive walk from a single inferred function to a human-in-the-loop service
served over HTTP that resumes after a restart. About fifteen minutes. For the minimal
example, see the [README](../README.md).

> These examples target the current repository implementation. Published packages
> may lag behind `main`; see [CONTRIBUTING.md](../CONTRIBUTING.md) for checkout setup.

## Install

The tutorial builds up to the CLI and HTTP integrations, so install their extras
alongside the provider:

```bash
pip install 'sefios[litellm,web,cli,fastapi,sqlite]' uvicorn
```

Set whatever credentials your model needs (LiteLLM reads provider env vars):

```bash
export OPENAI_API_KEY=sk-...
```

## 1. Your first inferred function

An `@infer` function is an abstract method whose implementer is an LLM: the
signature is the input/output contract, the docstring is the instruction, the body
is `...`. You run it inside a **session**, which gives it durability and a store.

<!-- example: tutorial-quickstart -->
```python
# quickstart.py
import asyncio

from pydantic import BaseModel
from sefios import SQLitePersistence, SessionScope, domain


class Summary(BaseModel):
    key_points: list[str]
    uncertainty: str


infer = domain("quickstart").infer

@infer
async def summarize(article: str) -> Summary:
    """Summarize the article for a technical audience; note key uncertainty."""
    ...


scope = SessionScope(
    model="gpt-4o",
    persistence=SQLitePersistence(),
)


async def main() -> None:
    async with scope.session(session_id="quickstart"):
        result = await summarize("Large language models ...")
        print(result.key_points)


if __name__ == "__main__":
    asyncio.run(main())
```

```bash
python quickstart.py
```

The body never runs. sefia sends the signature, docstring, and arguments to the
model, then validates the response into a `Summary`. `SessionScope` wired the LLM
client, the durability session, and a SQLite database under `.sefios/` for you. For the
full rules on arguments, service members, tools, and return types, see
[The `@infer` contract](./infer-contract.md).

Memory is the process-local default. This tutorial opts into SQLite so glyff execution
records, Sefia session state, and the session registry survive restarts in one database.
JSON files remain available for debugging with `FilePersistence` from the
`sefios[file-store]` extra.

By default, the provider constrains decisions with native structured output.
Providers without that capability can use `PromptedDecisionTransport`, which asks
for the same decision format in the prompt and validates it locally. Both transports
provide the same final results, repair behavior, and streaming features. Core also
provides `NativeDecisionTransport` for clients supporting native function calls, with
the same observable behavior.

## 2. Give it a tool

Tools are the **public methods of fields granted with the `Tools[...]` annotation**
— no decorator, no registry, no base class. Hold a dependency in a class-level field
annotated `Tools[...]`, and its public methods become callable by the inferred step.

Replace the `main` function in `quickstart.py` with the code below, keeping the
imports, models, `infer`, and `scope` above it. Keep the `if __name__` block last
so it calls the new `main`.

<!-- example: tutorial-tools -->
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

    @infer
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

<!-- example: tutorial-cli -->
```python
# hitl_cli.py
import asyncio
from pathlib import Path

import typer
from pydantic import BaseModel
from sefios import SQLitePersistence, Tools, domain
from sefios.cli import SefiaCLI
from sefios.sessions import FileActiveSessionStore
from sefios.tools import Input, WebSearch


class Report(BaseModel):
    topic: str
    summary: str


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


app = typer.Typer()
SESSION_DIR = Path(".sefios")
cli = SefiaCLI(
    model="gpt-4o",
    persistence=SQLitePersistence(),
    active_session_store=FileActiveSessionStore(SESSION_DIR / "active_session.txt"),
)
service = ResearchService(web=WebSearch(), input_tool=cli.input_tool)


@app.command()
def run(answer: str | None = None, interaction_id: str | None = None) -> None:
    async def _run() -> None:
        async with cli.session() as session:
            if answer is not None:
                if interaction_id is None:
                    raise typer.BadParameter("Provide --interaction-id from the pending request.")
                await session.resolve_interaction(interaction_id, answer)
            report = await service.run("the state of durable LLM applications")
            print("DONE:", report.summary)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
```

The SQLite provider makes session resources durable. `FileActiveSessionStore`
separately remembers which session this CLI workspace selected; both concerns default
to memory when omitted.

Run it once with no answer; it researches, drafts, then pauses:

```bash
python hitl_cli.py
# [INTERACTION_REQUIRED:<interaction_id>] {"type": "input", "prompt": "Approve the draft?"}
```

Now run it **again** (a fresh process) with the answer. The clarify/search/draft
steps are not re-run; they **replay their exact stored outputs**, so the model is
approving the *same* draft, and only the finalize step executes:

```bash
python hitl_cli.py --interaction-id "<interaction_id>" --answer "yes, approve"
# DONE: ...
```

Resume from the same working directory so the SQLite database and active-session
file are found. Another machine or deployment needs those same persisted resources
and stable domain/function identities; local files are not replicated automatically.
There was no checkpoint code, no step keys, no idempotency bookkeeping; just a tool
that raised and a session that replays.

## 4. Serve it over HTTP

The same service behind a stateless request/response handler. A pause returns
"needs input"; the input arrives in a later request to the same session id, and the
run resumes. Nothing runs in the background between the two requests.

<!-- example: tutorial-http -->
```python
# server.py
from fastapi import FastAPI
from pydantic import BaseModel, JsonValue
from sefios import SQLitePersistence
from sefios.fastapi import SefiaHTTP
from sefios.exceptions import InteractionRequired
from sefios.tools import WebSearch

from hitl_cli import ResearchService

app = FastAPI()
api = SefiaHTTP(
    model="gpt-4o",
    persistence=SQLitePersistence(),
)
research_service = ResearchService(web=WebSearch(), input_tool=api.input_tool)


class TurnBody(BaseModel):
    task: str
    interaction_id: str | None = None
    result: JsonValue = None


@app.post("/sessions")
def create_session():
    return {"session_id": api.create_session()}


@app.post("/sessions/{session_id}/turn")
async def turn(session_id: str, body: TurnBody):
    try:
        async with api.session(session_id=session_id) as session:
            if body.interaction_id is not None:
                await session.resolve_interaction(body.interaction_id, body.result)
            report = await research_service.run(body.task)
            return {"status": "done", "report": report}
    except InteractionRequired as e:
        return {"status": "interaction_required", "interaction_id": e.interaction_id, "request": e.request}
```

Pass `llm_client=` instead of `model=` when the HTTP integration should use a
custom `LLMClient`, such as a test double or a provider-specific adapter.

```bash
uvicorn server:app
```

For a uv project, use `uv run uvicorn server:app`. To load credentials from
`.env`, use `uv run --env-file .env uvicorn server:app`.

In another terminal in the same project directory (Bash/Zsh):

```bash
# Create a session
SESSION_ID=$(curl -fsS -X POST http://127.0.0.1:8000/sessions |
  python -c 'import json, sys; print(json.load(sys.stdin)["session_id"])')

# Start the research turn
curl -fsS -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/turn" \
  -H 'Content-Type: application/json' \
  -d '{"task": "the state of durable LLM applications"}'
```

If the response is `{"status":"interaction_required","interaction_id":"...","request":{...}}`, send the reply
below with the same session ID and `task`. To try resuming after a restart, stop
and restart the server from the same directory before sending the reply, keeping
its `.sefios/` database.

```bash
curl -fsS -X POST "http://127.0.0.1:8000/sessions/$SESSION_ID/turn" \
  -H 'Content-Type: application/json' \
  -d '{"task": "the state of durable LLM applications", "interaction_id": "<interaction_id>", "result": "yes, approve"}'
```

A completed turn returns `{"status":"done","report":{...}}`; if it asks for
more input, repeat the reply request with your next answer.

Keep `task` unchanged when resuming and send requests for a given session one at
a time; the facade does not serialize concurrent turns. Asking for approval in
an LLM instruction is not an enforced approval gate. For a mandatory gate, use
explicit application control flow as in [use case 02](./usecases/02-approval-gated-workflow.md).

The handler is an ordinary stateless endpoint. The durable run lives in the store
under `.sefios/`, not in the process, so killing and restarting the server
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

- Select another `PersistenceProvider`, or drop to `sefia.Session` for full
  control over the LLM client, policies, and middleware. Policies can wrap
  [attempts, steps, or durable decision generation](../DESIGN.md#middleware-control-scopes).
- Read [The `@infer` contract](./infer-contract.md) for the rules on arguments,
  service members, tool methods, and return types.
- Read [use case 01](./usecases/01-human-in-the-loop.md) to see this same turn
  hand-rolled, and exactly what the framework removed.
- Read [Design](../DESIGN.md) and the [FAQ](./faq.md) for the model and
  the tradeoffs.
- For long-horizon "resume in N days" flows, add an external scheduler that re-calls
  the endpoint — see the [timer note in the FAQ](./faq.md#what-about-long-running-waits--timers).


## Application-controlled input

Use `require_input` when Python control flow must always request an answer.
`Input.get_input` remains a model-dispatched tool; neither API calls the other.

```python
from sefios import domain, require_input

engrave = domain("approval_example").engrave

@engrave
async def approve_refund(order_id: str, amount: int) -> bool:
    answer = await require_input(f"Approve refund of {amount} for {order_id}? (yes/no)")
    return answer.strip().lower() == "yes"
```

Call this function inside `SessionScope.session()`, `SefiaHTTP.session()`, or
`SefiaCLI.session()`. On an
unanswered request it raises `InteractionRequired`; return its `interaction_id` and
`request` payload to the client. In the next session invocation, first call
`await session.resolve_interaction(interaction_id, answer)`, then invoke the
same workflow with the same arguments. A negative answer is still non-empty text:
validate it explicitly before executing the operation.

Pre-request queued input and implicit routing are no longer supported. Every
resolution must target an existing interaction ID. Input uses the same durable
Interaction channel as arbitrary tools; it is an adapter, not a routing subsystem.

## Externally resolved tools

Declare the expected result type at the requesting tool. The transport never
needs to import that type:

```python
from pydantic import BaseModel
from sefia import current_tool_call_id
from sefios import require_interaction


class WeatherResult(BaseModel):
    temperature: int


async def get_weather(city: str) -> WeatherResult:
    interaction_id = current_tool_call_id()
    if interaction_id is None:
        raise RuntimeError("This function requires model tool dispatch.")
    return await require_interaction(
        interaction_id,
        {"type": "weather", "arguments": {"city": city}},
        WeatherResult,
    )
```

After a pause, another request or process can discover and resolve interactions:

```python
async with api.session(session_id=session_id) as session:
    pending = await session.pending_interactions()
    await session.resolve_interaction(interaction_id, {"temperature": 21})
    result = await service.run(task)
```

Use the stable tool call ID for model tools. Application-controlled interactions
select their own stable identity; `require_input` derives it from its engraved
execution. `InteractionChannel` never generates IDs.

For a custom integration, construct the public channel from the same persistence
provider and session ID used by `SessionScope`. Resolution and discovery can run
outside the execution scope, including in a new process using the same durable store:

```python
from sefios import SQLitePersistence, SessionScope
from sefios.interactions import InteractionChannel

persistence = SQLitePersistence("sessions.sqlite3")
scope = SessionScope(model="gpt-4o", persistence=persistence)
channel = InteractionChannel(persistence.create_session_storage(session_id))
pending = await channel.pending()
await channel.resolve(interaction_id, {"temperature": 21})
async with scope.session(session_id=session_id):
    result = await service.run(task)
```

The scope's internal binding stays private. Inside an existing scope,
`InteractionChannel(get_session_storage())` is also supported, using the public
`get_session_storage` function from `sefios`.

`require_interaction` accepts `TypeForm[T]` result contracts: classes, `list[int]`,
`dict[str, WeatherResult]`, `WeatherResult | None`, `Literal`, `Annotated`, and
`TypedDict` forms retain their inferred result types. Validation happens when the
requester resumes, not when the transport stores JSON. Pyright 1.1.410 requires
`enableExperimentalFeatures = true` for these type expressions; the repository
enables it in `pyproject.toml`.

Input owns its request payload and `on_prompt_delta` preview callback. Request and
completion observation uses generic interactions; Input-specific lifecycle DTOs
and callbacks are no longer exposed. The generic CLI reporter renders each opaque
request as JSON, and reports arbitrary execution pauses without assuming input
is required.

An interaction persists request/result facts only; Glyff handles execution replay
and pause/resume. Repeated identical requests or results are idempotent; changed
values conflict. Results, including JSON null, are immutable. A result that fails
the requester's type validation raises Pydantic `ValidationError` on replay and
cannot be overwritten. See [how it works](how-it-works.md#human-in-the-loop-pause--raise-resume--re-invoke)
for storage and concurrency guarantees.
