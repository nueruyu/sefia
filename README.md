# Sefia

**S**tateless **F**unction **I**nference **A**gent

Sefia lets you write LLM-powered behavior as typed async Python functions.

The function signature defines the inputs, the return type defines the output,
and the docstring defines the instruction. The body can stay empty: at runtime,
Sefia asks the LLM to produce the result and validates it against the declared
type.

```python
import glyff
from glyff.stores import MemoryClient
from glyff.stores import MemorySessionStore as GlyffMemorySessionStore
from pydantic import BaseModel
from sefia import Session, infer
from sefia.pydantic.glyff_serialization import SefiaArgsHasher, SefiaSerializer
from sefia.stores import MemorySessionStore as SefiaMemorySessionStore
from sefia_litellm import LiteLLMClient


class Summary(BaseModel):
    key_points: list[str]
    uncertainty: str


@infer()
async def summarize(article: str) -> Summary:
    """
    Summarize the article for a technical audience.
    Include key claims and note important uncertainty.
    """
    ...


async def main(article: str):
    serializer = SefiaSerializer()
    client = MemoryClient()
    glyff_store = GlyffMemorySessionStore(client=client, serializer=serializer)
    sefia_store = SefiaMemorySessionStore(client=client, serializer=serializer)

    async with glyff.Session(
        id="demo",
        store=glyff_store,
        hasher=SefiaArgsHasher(),
    ) as gs:
        async with Session(
            llm_client=LiteLLMClient(model="gpt-4o"),
            glyff_session=gs,
            session_store=sefia_store,
        ):
            summary = await summarize(article)
            print(summary.key_points)
```

The call site stays ordinary Python:

```python
brief = await clarify_request(user_request)
sources = await research(brief)
article = await write_article(brief, sources)
```

Sefia can also collect tools from objects you pass around. That gives you a
small way to model dependencies without turning the workflow into a graph DSL
or making an agent object the main unit of work.

```python
from pydantic import BaseModel
from sefia import infer, tool


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
```

The body of `generate_report` is not executed. Sefia sends the signature,
docstring, arguments, and available tools to the LLM, then returns a structured
`Report`.

## Why Sefia

Pydantic AI, LangGraph, and Sefia all help build LLM applications. Sefia is for
the cases where the most natural shape of the code is still a Python call graph:
small typed steps, composed with normal `await` calls, with tools available when
a step needs them.

- **LLM steps are functions.** Use `@infer` on an async function or method. The
  signature and docstring are the contract.
- **Tools are methods.** Use `@tool` on methods that an inferred step may call.
  Tool discovery follows reachable Python objects instead of a separate registry.
- **Composition stays in Python.** Branching, loops, retries around a workflow,
  and helper functions can use normal Python control flow.
- **Durability is lightweight.** Sefia is backed by
  [glyff](https://github.com/nueruyu/glyff), so function calls can be
  checkpointed, paused, resumed, and replayed without starting from a workflow
  engine such as Temporal.
- **State is explicit.** Pass inputs in, return outputs out, and keep mutable
  state in stores when a tool or workflow actually needs it.

## Installation

```bash
pip install sefia
```

To use LiteLLM as an LLM provider adapter:

```bash
pip install sefia_litellm
```

Sefia depends on [glyff](https://github.com/nueruyu/glyff) for checkpointing and
resumption. For LiteLLM-backed provider support, install
[sefia_litellm](./packages/sefia_litellm).

## Core concepts

### `@infer`

`@infer` turns a typed async function into an LLM-backed call.

```python
@infer()
async def decide_next_action(state: AgentState) -> AgentDecision:
    """Decide the next action for the current state."""
    ...
```

The signature and docstring define the contract. The LLM:

1. Receives the docstring as instructions and the arguments as task input.
2. Calls available `@tool` methods when tools are reachable from `self`.
3. Returns a value matching the declared return type.

Dataclasses, Pydantic models, primitives, and `Serializable` instances are
supported. Sefia generates a structured output schema from the declared return
type and validates the response before returning it.

### `@tool`

`@tool` exposes a method to inferred calls on the same object graph.

```python
class Calculator:
    @tool
    async def add(self, a: int, b: int) -> int:
        """Return the sum of two integers."""
        return a + b
```

Tools are automatically discovered when an `@infer` method runs. The discovery
rules are deliberately small:

- Methods of the instance itself are scanned.
- Private attributes, such as `self._web`, are scanned recursively as toolkits.
- Methods starting with `_` are never exposed, even if marked with `@tool`.
- Name collisions raise `ToolConflictError` at runtime.

To expose only a subset of tools, return a narrower toolkit object:

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

This keeps Sefia's tool model simple: it exposes what is reachable. Scoping is a
property of the toolkit you provide.

### `Session`

`Session` wires up the LLM client, policies, and the underlying glyff session.
The glyff store records replay state; the Sefia store records Sefia-specific
metadata.

```python
async with glyff.Session(id="abc-123", store=glyff_store, hasher=hasher) as gs:
    async with Session(
        llm_client=llm,
        glyff_session=gs,
        session_store=sefia_store,
    ):
        result = await summarize(text)
```

### Policies

Policies attach event handlers that govern how the inference loop reacts to
errors and other events.

```python
from sefia import MaxRetries, infer


class MyAgent:
    @infer(policies=[MaxRetries(count=5)])
    async def critical_task(self, request: Request) -> Result:
        """Handle the request with retry behavior."""
        ...
```

Built-in policies include `MaxRetries` and `MaxTurns(count=...)`, which caps
the number of turns in a single inference loop (default 25). Custom policies
can be added by implementing the `Policy` ABC.

### Resumption and interruption

Sefia inherits glyff's replay model. In practice, this means a call can pause
while waiting for outside input, then resume later without re-running completed
engraved work.

```python
from glyff import engrave
from glyff.exceptions import YieldException
from sefia import tool


class HumanInputTool:
    @tool
    @engrave
    async def ask_user(self, question: str) -> str:
        """Ask the user a question and resume when an answer is available."""
        if answer := await find_answer_for_this_call():
            return answer

        await record_pending_question(question)
        raise YieldException()
```

A web handler can return `202 Accepted` with the session ID, then resume by
calling the same workflow again when the answer arrives.

## Event handlers

The inference loop emits events at each significant step. Custom handlers can
observe or alter behavior:

- `InferenceStart`: an `@infer` call begins
- `AttemptStart`: a new attempt within the retry loop
- `BeforeInferenceStep` and `AfterInferenceStep`: surrounding each LLM decision
- `BeforeToolCall`, `AfterToolCall`, and `ToolExecutionFailed`: surrounding tool
  execution
- `InferenceEnd`: the call completes

Handlers are typed by event:

```python
from sefia import EventHandler, Policy
from sefia.events import AfterInferenceStep


class DecisionLogger(EventHandler[AfterInferenceStep]):
    @property
    def event_types(self):
        return (AfterInferenceStep,)

    async def handle(self, event: AfterInferenceStep) -> None:
        print(type(event.decision).__name__)


class LoggingPolicy(Policy):
    def create_handlers(self) -> list[EventHandler]:
        return [DecisionLogger()]
```

Pass policies to `Session` or to a specific `@infer` call.

The built-in `StagnationDetector` is registered by default and aborts the loop
if the same tool is called repeatedly with identical arguments.

## Resources

`Resource[T]` is an abstract reference to a value that lives outside the LLM's
context. Use it when passing large objects between `@infer` functions.

```python
from pydantic import BaseModel
from sefia import Resource


class FileResource(Resource[bytes], BaseModel):
    path: str

    async def get(self) -> bytes: ...

    async def set(self, value: bytes) -> None: ...
```

The `T` parameter has no constraints. The resource itself must be serializable,
as a Pydantic model or `Serializable`, so it can be checkpointed by glyff.

## How it compares

These tools overlap. The difference is mostly in the mental model they ask you
to use.

| Tool        | Main shape                  | Good fit                                              |
| ----------- | --------------------------- | ----------------------------------------------------- |
| LangGraph   | Graphs, nodes, edges, state | Explicit state machines and complex routing           |
| Pydantic AI | Agent objects               | Typed agent applications centered on an agent runtime |
| Sefia       | Typed async functions       | LLM steps that read naturally as Python calls         |
| Temporal    | Distributed workflows       | Production workflow infrastructure across services    |

LangGraph is a good fit when the graph is the thing you want to see and operate.
Sefia keeps ordinary agent logic as a Python call graph.

Pydantic AI is a good fit when an `Agent` object is the center of the app. Sefia
starts from functions and methods, then lets tools and sessions sit around those
calls.

Temporal is excellent infrastructure when you need a distributed workflow
engine. Sefia and glyff cover a lighter part of the space: replaying Python
function calls for LLM workflows before that infrastructure is necessary.

## Status

Sefia is under active development. The API is not yet stable.

## License

MIT
