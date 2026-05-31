# sefia

**S**tateless **F**unction **I**nference **A**gent

Define LLM agents as plain Python classes. Let the LLM infer the implementation of methods from their signatures and docstrings.

```python
from pydantic import BaseModel
from sefia import infer, tool, Session
from sefia_litellm import LiteLLMClient


class Report(BaseModel):
    topic: str
    summary: str
    sources: list[str]


class WebToolkit:
    @tool
    async def search(self, query: str) -> list[str]:
        """Search the web and return relevant URLs."""
        ...

    @tool
    async def fetch_content(self, url: str) -> str:
        """Fetch the text content of the given URL."""
        ...


class Researcher:
    def __init__(self, web: WebToolkit):
        self._web = web

    @infer()
    async def generate_report(self, topic: str) -> Report:
        """Research the topic using the web tools and produce a structured report."""
        ...


async def main():
    async with Session(llm_client=LiteLLMClient(model="gpt-4o")) as session:
        researcher = Researcher(WebToolkit())
        report = await researcher.generate_report(topic="sefia")
        print(report.summary)
```

The body of `generate_report` is never executed. Instead, sefia hands the signature, docstring, and available tools to an LLM, which decides what to do and returns a structured result.

## Why sefia

Most LLM agent frameworks ask you to learn a new mental model: graphs, configurations, role definitions, or a domain-specific orchestration layer. sefia takes a different approach.

- **Agents are classes, behaviors are methods.** No graphs, no YAML, no role descriptions in markdown files. Just Python.
- **Function inference is the core abstraction.** A method decorated with `@infer` declares a spec; the LLM provides the implementation at runtime.
- **Tools are just methods.** Any method decorated with `@tool` becomes available to the LLM. No registries, no manual exposure.
- **Stateless by design.** Agents hold their dependencies, not their state. Pass arguments in, get results out. Easier to test, easier to reason about, easier to resume.
- **Resumable across transports.** Backed by [glyff](../glyff), every method call is a checkpoint. Interrupt during user input, an API failure, or a process crash — resume from the same point.

## Installation

```bash
pip install sefia
```

To use LiteLLM as an LLM provider adapter:

```bash
pip install sefia_litellm
```

sefia depends on [glyff](../glyff) for checkpointing and resumption. For LiteLLM-backed provider support, install [sefia_litellm](./packages/sefia_litellm).

## Core concepts

### `@infer` — let the LLM implement the method

```python
@infer()
async def summarize(self, text: str) -> str:
    """Produce a concise summary of the given text."""
    ...
```

The signature and docstring define the contract. The LLM:

1. Receives the docstring as system instructions and the arguments as the user prompt.
2. Calls any `@tool` methods reachable from `self` as needed.
3. Returns a value matching the declared return type. Pydantic models, primitives, and `Serializable` instances are all supported.

If the return type is a Pydantic model, sefia automatically requests structured output and validates the response.

### `@tool` — expose a method to the LLM

```python
class Calculator:
    @tool
    async def add(self, a: int, b: int) -> int:
        """Return the sum of two integers."""
        return a + b
```

Tools are automatically discovered when an `@infer` method runs. The discovery rules are simple:

- Methods of the agent instance itself are scanned.
- Private attributes (those starting with `_`) of the agent are scanned recursively as toolkits.
- Methods starting with `_` are never exposed, even if marked with `@tool`.
- Name collisions raise `ToolConflictError` at runtime.

To expose only a subset of tools, give the toolkit its own method that returns a narrower interface:

```python
class FullToolkit:
    @tool
    async def read(self, path: str) -> str: ...
    @tool
    async def write(self, path: str, data: bytes) -> None: ...

    def read_only(self) -> "ReadOnlyToolkit":
        return ReadOnlyToolkit()


class ReadOnlyToolkit:
    @tool
    async def read(self, path: str) -> str: ...
```

This keeps sefia's tool model simple: it always exposes everything reachable. Scoping is a concern of the toolkit, not the framework.

### `Session` — set up the execution context

```python
async with Session(llm_client=LiteLLMClient(model="gpt-4o")) as session:
    result = await agent.execute(...)
```

`Session` wires up the LLM client, the event handlers, and the underlying glyff session that provides checkpointing. If you already have a glyff session, pass it in:

```python
async with glyff.Session(session_id="abc-123") as gs:
    async with Session(llm_client=llm, glyff_session=gs) as ss:
        result = await agent.execute(...)
```

### Policies — control retry and recovery behavior

Policies attach event handlers that govern how the inference loop reacts to errors and other events.

```python
from sefia import infer, MaxRetries

class MyAgent:
    @infer(policies=[MaxRetries(count=5)])
    async def critical_task(self, ...) -> ...:
        """..."""
        ...
```

Built-in policies include `MaxRetries`. Custom policies can be added by implementing the `Policy` ABC.

### Resumption and interruption

sefia inherits glyff's interruption model. Any tool can raise `SessionInterrupted` to halt the session, and a subsequent call with the same session ID resumes from where it stopped.

```python
class UserConfirmTool:
    def __init__(self, http_session):
        self._session = http_session

    @tool
    async def ask_user(self, question: str) -> str:
        """Ask the user a question and return their response."""
        if answer := await self._session.get_pending_answer():
            return answer
        await self._session.record_question(question)
        raise SessionInterrupted()
```

The HTTP handler around this can return `202 Accepted` with the session ID, then resume by replaying the same agent method when the answer arrives.

## Event handlers

The inference loop emits events at each significant step. Custom handlers can observe or alter behavior:

- `InferenceStart` — an `@infer` call begins
- `AttemptStart` — a new attempt within the retry loop
- `BeforeLLMCall`, `AfterLLMCall` — surrounding each LLM request
- `BeforeToolCall`, `AfterToolCall`, `ToolError` — surrounding each tool invocation
- `InferenceEnd` — the call completes

Handlers are typed by event:

```python
from sefia import EventHandler
from sefia.events import AfterLLMCall

class TokenLogger(EventHandler[AfterLLMCall]):
    async def handle(self, event: AfterLLMCall) -> None:
        if event.response.usage:
            print(f"Tokens used: {event.response.usage}")
```

Register handlers at the session level:

```python
async with Session(llm_client=llm, handlers=[TokenLogger()]) as session:
    ...
```

The built-in `StagnationDetector` is registered by default and aborts the loop if the same tool is called repeatedly with identical arguments.

## Resources

`Resource[T]` is an abstract reference to a value that lives outside the LLM's context. Use it when passing large objects between `@infer` methods.

```python
from sefia import Resource

class FileResource(Resource[bytes], BaseModel):
    path: str

    async def get(self) -> bytes: ...
    async def set(self, value: bytes) -> None: ...
```

The `T` parameter has no constraints. The resource itself must be serializable (Pydantic model or `Serializable`) so it can be checkpointed by glyff.

## How it compares

- **CrewAI / Agno / AutoGen**: framework-driven role definitions and orchestration. sefia is closer to plain Python — the framework gets out of the way.
- **PydanticAI**: similar in spirit, but sefia adds checkpointing and resumption via glyff.
- **LangGraph**: graph-based state machines. sefia stays at the level of methods and classes.
- **Temporal**: a full distributed workflow engine. sefia and glyff cover the lightweight, in-process portion of that problem space, designed specifically for LLM agents.

## Status

sefia is under active development. The API is not yet stable. See [the design document](./docs/design.md) for the underlying decisions and trade-offs.

## License

TBD
