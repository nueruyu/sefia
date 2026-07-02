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

Avoid passing live application objects as arguments: clients, database sessions,
open files, sockets, caches, or service objects. Put those on a service as
dependencies instead, and expose narrow tool methods when the model needs to use
them.

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

## Service members

`self` is not sent to the model as an argument. Instance attributes are for
application dependencies: tools, stores, clients, configuration, or other service
objects.

If the model needs to see data, pass it as an argument. If the model needs to do
something, expose a tool through a held dependency.

A service class is also a capability boundary. Multiple `@infer` methods on the
same service share the tool surface collected from that service and its held
dependencies. Split services when different operations need different tools,
different write permissions, or unrelated capabilities.

### `self` and replay identity

Although `self` is not shown to the model, it still participates in the engraved
call identity. For a method call, the durable execution key is based on the method
identity and the bound arguments, including `self`.

That means two service instances can be distinct for replay even when the visible
task arguments are the same. With the default Glyff/Pydantic hasher, structured
values such as Pydantic models and dataclasses are hashed by value, while opaque
objects fall back to a qualified name representation. If an instance must be
replay-distinct by tenant, user, store, or other runtime identity, include that
identity in a stable field or argument rather than relying on object identity.

## Tool methods

Tool parameters must have type annotations. Prefer explicit parameters over
`*args` or `**kwargs`; only explicit positional or keyword parameters become part
of the tool schema. Defaults are allowed. `self` and `cls` are not tool
parameters.

Tool return values should also be structured values the model can read: strings,
numbers, booleans, lists, dictionaries, Pydantic models, dataclasses, or other
serializable data.

## Practical rules

- Put task input in function arguments.
- Put application capabilities on service dependencies.
- Keep return types explicit and structured.
- Keep tool parameters explicit and typed.
- Remember that `self` affects replay identity even though it is not prompt input.
- Split services when a tool should not be visible to every inferred method on
  that service.
