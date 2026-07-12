# The `@infer` contract

An `@infer` function is an ordinary Python function or method whose body is not
executed. Its signature, type hints, return annotation, and docstring are the
contract used to run an LLM-backed call.

## Function shape

Use explicit parameters, an explicit return type, and a docstring instruction.

```python
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

Avoid passing live application objects as **data** arguments: clients, database
sessions, open files, sockets, caches, or service objects. When the model needs to
*use* one, pass it as a **capability parameter** instead — a parameter whose declared
type bears the `Tools` role (see *Tools and capability parameters* below) — or hold
it as a dependency on a service. A capability parameter's methods become tools; it is
not rendered into the prompt as data.

Treat data arguments as replay inputs. Prefer stable values — IDs, paths, URLs, text,
or structured data — over mutable objects whose meaning can change between
invocations. (Capability parameters, like `self`, still participate in the engraved
call identity; see below.)

## Return types

Return types should be Pydantic-schema-compatible: primitives, Pydantic models,
dataclasses, lists, dictionaries, unions, optionals, literals, enums, and other
typing constructs Pydantic can describe and validate.

The model returns JSON, and sefia validates the final answer against the declared
return type before returning it to Python.

Use `Never` only for tool-only inferred functions. A `Never`-returning function
must have tools available, because it can never finish with a final answer.

## Tools and capability parameters

A method becomes a tool only when it is reachable through a `Tools`-bearing declared
type, starting from a **capability parameter**. There is no ambient authority:
holding an object is not enough — its declared type must carry the role.

- **`self`/`cls`** are capability parameters by convention: the collector scans the
  instance's held fields and exposes those whose **class-level declared type** bears
  `Tools`. Give held dependencies a class-level annotation (a dataclass field, or an
  explicit `_web: WebToolkit` in the class body) — an `__init__`-only assignment is
  not a declaration and exposes nothing.
- **Any other parameter** is a capability parameter only if its declared type bears
  the role — so a plain function gets tools too:
  `async def run(kit: WebToolkit, topic: str)`.
- **Mark a type with `Tools`** to make its public methods callable:
  `class WebToolkit(Tools): ...`. For a type you cannot edit, mark the field:
  `_web: Annotated[VendorClient, Tools]`.

`self` is not sent to the model as data. Instance attributes are for application
dependencies — tools, stores, clients, configuration — and a non-tool dependency
(a config, a store) simply doesn't bear `Tools`, so it never leaks.

### Narrowing a method's surface

Annotate `self` with a role-bearing `Protocol` to shape or restrict one method's
tools:

```python
class ResearchTools(Tools, Protocol):     # re-inherit Protocol, or it turns concrete
    @property
    def _web(self) -> ReadOnlyWeb: ...     # re-narrow a held field (read-only property;
                                           #   a plain attribute is invariant and won't
                                           #   type-check against a different concrete type)
    async def _score(self, url: str) -> float: ...   # opt the instance's own method in

class ResearchService:
    _web: WebToolkit
    _config: AppConfig                     # not in the protocol -> never exposed
    async def _score(self, url: str) -> float: ...
    @infer
    async def run(self: ResearchTools, topic: str) -> Report: ...
```

A service class is a capability boundary. Multiple `@infer` methods on the same
service share the tool surface from its held dependencies unless a method narrows its
own `self`. Split services (or annotate `self`) when operations need different tools
or different write permissions.

### Capability parameters and replay identity

Like `self`, a capability parameter participates in the engraved call identity even
though it is not prompt data. With the default Glyff/Pydantic hasher a dataclass or
Pydantic value is hashed by value, while an opaque object falls back to a qualified
name. If an instance must be replay-distinct by tenant/user/store, include that
identity in a stable field or argument rather than relying on object identity.

## Tool methods

A tool method is discovered on a `Tools`-bearing type as a plain function,
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
`@engrave` and similar `functools.wraps`-based decorators are fine.

## Practical rules

- Put task input in function arguments; put capabilities behind `Tools`-bearing types.
- Mark toolkits with `Tools` (`class WebToolkit(Tools)`), or a field with
  `Annotated[T, Tools]` for a type you can't edit.
- Give held dependencies a class-level annotation (a dataclass field is ideal) — an
  `__init__`-only assignment declares nothing and exposes nothing.
- Keep return types explicit and structured; keep tool parameters explicit and typed.
- Remember that capability parameters (including `self`) affect replay identity even
  though they are not prompt input.
- Narrow `self` with a surface `Protocol`, or split services, when a tool should not
  be visible to every inferred method on that service.
