# The `@infer` contract

An `@infer` function is an ordinary Python function or method whose body is not
executed. Its signature, type hints, return annotation, and docstring are the
contract used to run an LLM-backed call.

The snippets on this page illustrate individual contracts, not standalone scripts;
`Summary`, `Report`, and toolkit/config types stand for application-defined types.
For complete imports, implementations, and session setup, see the [tutorial](./tutorial.md).

## Function shape

Use explicit parameters, an explicit return type, and a docstring instruction.

```python
from sefios import domain

infer = domain("myapp").infer

@infer
async def summarize(article: str) -> Summary:
    """Summarize the article and return a structured summary."""
    ...
```

When the function is called, sefia binds the arguments, applies Python defaults,
and sends the non-`self` arguments to the model. The original function body is
ignored.

Prefer explicit return annotations. If no return type is declared, the output
contract becomes `Any`.

## Arguments

Arguments should be task data: text, identifiers, numbers, booleans, structured
data, Pydantic models, dataclasses, collections, or other values your configured
serializer can normalize.

Avoid passing live application objects as arguments: clients, database sessions,
open files, sockets, caches, or service objects. When the model needs to *use* one,
hold it on the service in a `Tools[...]`-granted field instead (see *Tools and
granted fields* below) — tool dependencies are expressed through classes, and a
plain `@infer` function has no tools.

Treat arguments as replay inputs. Prefer stable values — IDs, paths, URLs, text,
or structured data — over mutable objects whose meaning can change between
invocations.

## Return types

Return types should be Pydantic-schema-compatible: primitives, Pydantic models,
dataclasses, lists, dictionaries, unions, optionals, literals, enums, and other
typing constructs Pydantic can describe and validate.

The model returns JSON, and sefia validates the final answer against the declared
return type before returning it to Python.

Use `Never` only for tool-only inferred functions. A `Never`-returning function
must have tools available, because it can never finish with a final answer.

## Tools and granted fields

A method becomes a tool only when its holder is **granted** with the `Tools` alias
in a class-level field annotation — holding an object is not enough:

```python
class ResearchService:
    _web: Tools[WebToolkit]     # granted: WebToolkit's public methods are tools
    _config: AppConfig          # plain dependency: never exposed

    def __init__(self, web: WebToolkit, config: AppConfig):
        self._web = web
        self._config = config

    @infer
    async def run(self, topic: str) -> Report: ...
```

- `Tools[T]` is an `Annotated` alias: checkers treat it as `T`, `T` stays a plain
  class or `Protocol`, and it composes with `Optional`, other `Annotated` metadata,
  dataclass `field(...)`, and defaults.
- The grant must be a **class-level annotation** (a bare class-body annotation is
  enough) — an `__init__`-only assignment is not a declaration and exposes nothing.
- Narrow a broad object by granting through a protocol: `_web: Tools[ReadOnlyWeb]`
  exposes only the protocol's declared members.
- Tools ride on the `@infer` method's receiver (`self`); every other parameter is
  task data, and a plain `@infer` function has no tools.

### Selecting a method's surface

Annotate `self` with a plain `Protocol` to select one method's tools; the annotation
itself is the opt-in, and it replaces the class-body grants for that method:

```python
class ResearchSurface(Protocol):
    @property
    def _web(self) -> ReadOnlyWeb: ...   # re-narrow a held field
    async def _score(self, url: str) -> float: ...   # opt the own method in

class ResearchService:
    _web: Tools[WebToolkit]
    _config: AppConfig                   # not in the protocol -> never exposed
    async def _score(self, url: str) -> float: ...
    @infer
    async def run(self: ResearchSurface, topic: str) -> Report: ...
```

Re-narrowed fields must be declared as read-only properties: a plain protocol
attribute is invariant and will not type-check against a different concrete type.
The surface is granted exactly as declared — including the running `@infer` method
itself, if you declare it. Only declare it when self-recursion is intended.

A service class is a capability boundary. Multiple `@infer` methods on the same
service share the granted fields unless a method selects its own surface. Split
services (or annotate `self`) when operations need different tools or different
write permissions.

### `self` and replay identity

`self` participates in the engraved call identity even though it is not prompt
data. With the default Glyff/Pydantic canonicalizer a dataclass or Pydantic value is
represented by value, while an opaque object falls back to its qualified type name. If an
instance must be replay-distinct by tenant/user/store, include that identity in a
stable field or argument rather than relying on object identity.

## Tool methods

A tool method is discovered on a granted field's declared type as a plain function,
`staticmethod`, or `classmethod`. Tool parameters must have type annotations. Prefer
explicit parameters over `*args`/`**kwargs`; only explicit positional or keyword
parameters become part of the tool schema. Defaults are allowed. `self` and `cls`
are not tool parameters.

Tool return values should also be structured values the model can read: strings,
numbers, booleans, lists, dictionaries, Pydantic models, dataclasses, or other
serializable data.

A decorator applied to a tool method must return a function (use `functools.wraps`);
one that returns a non-function callable — a class-based wrapper or a bare
`functools.partial` — makes the method invisible to discovery, with no error.
Domain-bound engrave decorators and similar `functools.wraps`-based decorators are fine.

## Persistent execution identity

Use a domain-bound `infer` decorator for persisted application boundaries. Without a
`name`, it uses the function's module-qualified name; pass `name=...` when that
derived name must not follow a refactor. The domain id and execution name are storage
contracts. Keep them stable across refactors. Domains start at version `"1"`; pass
`version=` when migrating an existing domain to a new contract, then use Glyff's
domain migration API to remap existing sessions. Sefia's internal `inference.step`
and `inference.tool_calls` records belong to `sefia`, so application migrations do
not need to know their shapes.

An identity migration does not migrate completed result payloads or Sefia history
metadata. Keep readers backward-compatible, discard and re-run incompatible records,
or migrate those payloads separately.

## Practical rules

- Put task input in function arguments; grant capabilities with `Tools[...]` field
  annotations on the service.
- The grant must be class-level (a bare class-body annotation is enough) — an
  `__init__`-only assignment declares nothing and exposes nothing.
- Keep return types explicit and structured; keep tool parameters explicit and typed.
- Remember that `self` affects replay identity even though it is not prompt input.
- Select a method's surface with a `self:` `Protocol` annotation, or split services,
  when a tool should not be visible to every inferred method on that service.
