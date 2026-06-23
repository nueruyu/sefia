# Sefia

**S**tateless **E**ngraved **F**unction **I**nference **A**bstraction

Sefia lets you define LLM-powered behavior as typed async Python functions.

The function signature defines the inputs, the return type defines the output,
and the docstring defines the instruction.

```python
import glyff
from glyff.store import MemoryClient
from glyff.store import MemorySessionStore as GlyffMemorySessionStore
from glyff_pydantic import PydanticArgsHasher, PydanticSerializer
from pydantic import BaseModel
from sefia import Session, infer
from sefia.stores import MemorySessionStore as SefiaMemorySessionStore
from sefia_litellm import LiteLLMClient


class Summary(BaseModel):
    key_points: list[str]
    uncertainty: str


@infer
async def summarize(article: str) -> Summary:
    """
    Summarize the article for a technical audience.
    Include key claims and note important uncertainty.
    """
    ...


async def main(article: str):
    serializer = PydanticSerializer()
    client = MemoryClient()
    glyff_store = GlyffMemorySessionStore(client=client, serializer=serializer)
    sefia_store = SefiaMemorySessionStore(client=client, serializer=serializer)

    async with glyff.Session(
        id="demo",
        store=glyff_store,
        hasher=PydanticArgsHasher(),
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

    @infer
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
- **Tools are opt-in methods.** Mark a method with `@tool` to make it callable
  by an inferred step. Discovery follows the Python objects reachable from
  `self` instead of a separate registry. For classes you cannot decorate, wrap
  them with `toolify()`.
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
@infer
async def decide_next_action(state: AgentState) -> AgentDecision:
    """Decide the next action for the current state."""
    ...
```

The signature and docstring define the contract. The LLM:

1. Receives the docstring as instructions and the arguments as task input.
2. Calls `@tool` methods when tools are reachable from `self`.
3. Returns a value matching the declared return type.

Dataclasses, Pydantic models, primitives, and standard typing constructs are
supported. Sefia generates a structured output schema from the declared return
type and validates the response before returning it.

#### `-> Never`: tool-only loops

Annotate an `@infer` function with `-> Never` when it should never produce a
final answer — only call tools, indefinitely.

```python
from typing import Never
from sefia import infer


class ChatAgent:
    @infer
    async def chat(self) -> Never:
        """
        Have a conversation with the user via HumanInputTool.
        Always call the tool to get the next message; never return a final answer.
        """
        ...
```

When the return type is `Never`, Sefia enforces tool-only execution at every
layer: the JSON schema sent to the LLM omits the `final_answer` field entirely,
the system prompt instructs the model to always use `tool_calls`, and the
executor raises `RuntimeError` if a final answer somehow arrives anyway. The
inference loop then runs until an external signal (such as `YieldException` from
a human-input tool) interrupts it.

### `@tool`

Tools are opt-in. Mark a method with `@tool`, and when an `@infer` step runs,
Sefia discovers the marked methods reachable from `self` and offers them to the
LLM. The discovery rules are deliberately small:

- **The instance's own `@tool` methods** are exposed, including private
  (`_`-prefixed) ones — a step can call its own marked helpers.
- **`@infer` methods are not tools** unless you also mark them. A bare `@infer`
  entry point never exposes itself, so the running step cannot recurse into
  itself by accident.
- **Dependencies held in attributes**, public or private (such as `self._web`
  or `self.calculator`), contribute their `@tool` methods.
- Name collisions raise `ToolConflictError` from `sefia.exceptions` at runtime.

```python
class Calculator:
    @tool
    async def add(self, a: int, b: int) -> int:
        """Return the sum of two integers."""
        return a + b


class MathAgent:
    def __init__(self, calculator: Calculator):
        self._calculator = calculator  # its @tool methods become tools

    @infer
    async def solve(self, problem: str) -> int:
        """Solve the problem, using the calculator when arithmetic is needed."""
        ...
```

To expose an inferred step of one agent as a tool for another, stack the
decorators — `@tool` over `@infer`:

```python
class Researcher:
    @tool
    @infer
    async def research(self, topic: str) -> list[str]:
        """Research the topic and return supporting URLs."""
        ...
```

### `toolify`

`@tool` works for your own classes, but you often want to expose a third-party
client or a plain function you cannot decorate. `toolify()` bundles objects and
functions into a `Toolset` you hold like any other dependency:

```python
from sefia import infer, toolify


async def current_time() -> str:
    """Return the current time as an ISO-8601 string."""
    ...


class Assistant:
    def __init__(self, client: SomeExternalClient):
        # Every public method of `client`, plus the standalone function.
        self._tools = toolify(client, current_time)

    @infer
    async def handle(self, request: str) -> str:
        """Handle the request using the available tools."""
        ...
```

`toolify(obj)` exposes every public callable method of `obj` (its `_`-prefixed
helpers stay private) and registers any function passed directly. This is
convenient, but broad external clients can contain methods you do not want the
LLM to call. For production use, prefer a narrow adapter object and
`toolify(adapter)`, or pass only the specific functions you intend to expose.

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

A policy contributes two kinds of extension to an inference run, and may provide
either or both:

- **Observation** via `create_handlers()` — event handlers that are notified of
  events but cannot steer the loop (see [Event handlers](#event-handlers)).
- **Control** via `create_middleware()` — middleware that wraps the run (or each
  step) and steers the executor's loops.

```python
from sefia import infer, policy
from sefios.policies import MaxRetries


class MyAgent:
    @infer
    @policy(MaxRetries(count=5))
    async def critical_task(self, request: Request) -> Result:
        """Handle the request with retry behavior."""
        ...
```

A per-function policy is attached with the separate `@policy` decorator. It
records the policy under the `"policies"` key of the function's
`__sefia_metadata__`, where `@infer` reads it — the order of the two
decorators does not matter. To apply more than one policy, merge them on the
caller side (or stack multiple `@policy` decorators).

Built-in policies include `MaxRetries` and `MaxSteps`. Custom policies can be
added by implementing the `Policy` ABC.

### Profiles

A `Profile` is a keyed, reusable bundle of inference configuration: the model
(its `LLMClient`) plus any policies that should apply whenever it is selected.
Build profiles up front, register them on the `Session`, and select one per
function with `@profile`:

```python
from enum import Enum, auto

from sefia import Profile, Session, infer, profile
from sefia_litellm import LiteLLMClient
from sefios.policies import MaxRetries


class Models(Enum):
    FAST = auto()
    SMART = auto()


class MyAgent:
    @infer
    async def quick_step(self, request: Request) -> Draft:
        """Runs on the session default model and policies."""
        ...

    @infer
    @profile(Models.SMART)
    async def hard_step(self, draft: Draft) -> Result:
        """Runs on the SMART profile instead."""
        ...


async with Session(
    llm_client=LiteLLMClient(model="gpt-4o-mini"),
    glyff_session=gs,
    session_store=sefia_store,
    profiles=[
        Profile(
            key=Models.SMART,
            client=LiteLLMClient(model="gpt-4o"),
            policies=[MaxRetries(count=5)],
        )
    ],
):
    ...
```

A profile `key` is **any hashable value**, not just a string — an `Enum` member
(shown above), an `int`, or a plain `"smart"` all work — so you can avoid
stringly-typed configuration. `@profile` records the key under the `"profile"`
slot of the function's `__sefia_metadata__` (just like `@policy`), so its order
relative to `@infer` does not matter. Selecting by key keeps the call site
decoupled from the concrete client — a test can bind the same key to a mock. An
unknown key fails fast at call time with the list of registered profiles.

Configuration is **layered, most specific wins**:

```text
function (@policy / @profile)  >  profile  >  session
```

- **Model/client:** the selected profile's `client` overrides the session's
  default `llm_client`; with no `@profile`, the session default is used.
- **Policies:** additive across layers and collected most-general first
  (`session → profile → function`), so a function's own `@policy` decorators sit
  closest to the call.

Model settings (temperature, max tokens, ...) ride on how the profile's client
is constructed today; the profile is the seam where first-class settings can be
added later.

### Steps and the inference loop

The executor owns the inference lifecycle and the inner step loop, and wraps
each unit of work with the configured middleware. Inference middleware can
retry by calling its wrapped function again, while step middleware can stop the
step loop by raising an exception.

Two seams are available, exposed as ABCs from `sefia`:

- `InferenceMiddleware.wrap(ctx, nxt)` wraps a whole inference run. `Retrier`
  uses this: on a recoverable `InferenceError` it calls `nxt` again, and once
  the budget is spent it re-raises the original error — which, being a
  `YieldException`, propagates as a resumable yield rather than a hard failure
  (see [Recoverable inference errors](#recoverable-inference-errors)).
- `StepMiddleware.wrap(ctx, nxt)` wraps a single step. `MaxSteps` uses this to
  cap the loop, raising `MaxStepsExceededError` once the step limit is reached.

The executor does not cap the loop on its own; without a `StepMiddleware` there
is no default cap. The `sefios` session helpers add `MaxSteps(count=25)` by
default; pass `max_steps=None` to opt out.

```python
from sefia import infer, policy
from sefios.policies import MaxSteps


class MyAgent:
    @infer
    @policy(MaxSteps(count=25))
    async def run(self, task: Task) -> Result:
        """Work the task, capped at 25 steps."""
        ...
```

`MaxSteps` can also be passed to `Session(policies=[...])` to apply across an
entire session.

A failing tool is never treated as a retryable failure: the executor stringifies
the error into the history and feeds it back to the model so it can recover,
rather than restarting the run.

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

#### Recoverable inference errors

A recoverable inference failure — a transient provider hiccup (timeout, rate
limit, connection, temporarily unavailable) or an LLM response that fails schema
validation — is treated the same way. `InferenceError` subclasses
`YieldException`, so glyff never engraves it as a permanent `FAILED` record;
instead the step is left resumable. Re-invoking the workflow re-runs just that
step, and earlier completed steps replay without re-running. This means a
one-off network blip or malformed model response never poisons a run.

The `@infer` call surfaces the failure as the typed error, which you can catch
either as an `InferenceError` or as a `YieldException`. The abstract
`InferenceError` and `InvalidInferenceResponseError` live in `sefia`; a client
adapter contributes its own provider-shaped subclasses (for example
`sefia_litellm` defines `InferenceTimeoutError`, `InferenceRateLimitError`, and
friends). Genuinely permanent failures (authentication, malformed request,
content policy, ...) are *not* mapped to `InferenceError`; they propagate as
their own exceptions and are engraved as genuine failures.

`Retrier` adds an in-process fast path on top of this: it retries a recoverable
error a few times within the same process, and only once that budget is spent
does it let the error propagate as a resumable yield.

## Event handlers

The inference loop emits events at each significant step. Handlers **observe**
these events — they do not steer the loop. A handler's exception is logged and
swallowed so a misbehaving observer can never break the core loop. (Control
lives in middleware instead; see [Steps and the inference
loop](#steps-and-the-inference-loop).)

- `InferenceStart`: an `@infer` call begins
- `AttemptStart`: a new inference attempt
- `BeforeInferenceStep` and `AfterInferenceStep`: surrounding each LLM decision
- `BeforeToolCall`, `AfterToolCall`, and `ToolExecutionFailed`: surrounding tool
  execution
- `InferenceEnd`: the call completes

Handlers are typed by event. `EventHandler[SomeEvent]` infers the subscribed
event type automatically; `EventHandler[A | B]` or `EventHandler[Union[A, B]]`
subscribes to multiple event types.

```python
from sefia import Policy
from sefia.event_system import EventHandler
from sefia.events import AfterInferenceStep


class DecisionLogger(EventHandler[AfterInferenceStep]):
    async def handle(self, event: AfterInferenceStep) -> None:
        print(type(event.decision).__name__)


class LoggingPolicy(Policy):
    def create_handlers(self) -> list[EventHandler]:
        return [DecisionLogger()]
```

Pass policies to `Session` or to a specific `@infer` call.

Use `sefios.policies.StagnationPolicy` when you want to abort the loop if the
same tool is called repeatedly with identical arguments.

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
