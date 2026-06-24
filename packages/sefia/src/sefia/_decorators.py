import functools
import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Concatenate,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    overload,
)

from glyff import engrave

from . import _metadata
from ._context import get_context
from ._executor import InferenceExecutor
from ._interfaces import InferenceMiddleware, Policy, StepMiddleware
from .event_system import EventPublisher

C = TypeVar("C", bound=Callable[..., object])
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")
_R_co = TypeVar("_R_co", covariant=True)
_StreamH = TypeVar("_StreamH", bound=Callable[..., Awaitable[None]])


class _PolicyDecorator(Protocol):
    """Callable that decorates a function without changing its type."""

    def __call__(self, func: C) -> C: ...


class StreamableTool(Protocol[P, _R_co]):
    """A ``@tool``-marked method that also exposes a ``.stream`` decorator.

    Accessing it on an instance binds ``self`` (via ``__get__``), so the tool is
    called with its normal signature; accessing ``.stream`` on the marker itself
    registers a side-channel handler for the tool's streamed arguments.
    """

    def stream(self, handler: _StreamH, /) -> _StreamH:
        """Register a handler that receives this tool's streamed arguments."""
        ...

    def __get__(
        self, instance: object, owner: type | None = None, /
    ) -> Callable[P, _R_co]: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> _R_co: ...


@overload
def tool(func: Callable[Concatenate[Any, P], R]) -> StreamableTool[P, R]: ...
@overload
def tool(func: T) -> T: ...
def tool(func: Any) -> Any:
    """
    Mark a method as a tool available to an @infer step.

    This is a pure marker: the inference executor normalizes sync/async return
    values, so the wrapped function is returned unchanged to preserve its
    signature for schema generation.

    The returned tool exposes a ``.stream`` decorator to register a side-channel
    handler that receives the tool's arguments incrementally as the model emits
    them (see :mod:`sefia.streaming`)::

        @tool
        async def ask_human(self, question: str) -> str: ...

        @ask_human.stream
        async def _(self, events): ...  # async for ev in events: ...

    The handler is best-effort live preview; the tool still runs with the fully
    decoded arguments.
    """
    # When @tool is applied over @classmethod/@staticmethod, mark the underlying
    # function — those descriptor objects may reject attribute assignment.
    target = func.__func__ if isinstance(func, (classmethod, staticmethod)) else func
    setattr(target, "__sefia_tool__", True)

    def _register_stream(handler: _StreamH) -> _StreamH:
        setattr(target, "__sefia_stream_handler__", handler)
        return handler

    setattr(target, "stream", _register_stream)
    if func is not target:
        try:
            setattr(func, "stream", _register_stream)
        except AttributeError:
            pass
    return func


def policy(p: Policy) -> _PolicyDecorator:
    """
    Decorator that attaches an inference policy to an ``@infer`` function.

    The policy is recorded under the ``"policies"`` key of the function's
    ``__sefia_metadata__`` dict, where ``@infer`` reads it. The order relative
    to ``@infer`` does not matter::

        @infer
        @policy(CustomPolicy(middleware=lambda: [Retrier(max_retries=5)]))
        async def step(...): ...

    To apply more than one policy, merge them on the caller side (or stack
    multiple ``@policy`` decorators).

    Two constraints follow from storing the policy on the innermost function:

    - Any decorator placed between ``@policy`` and ``@infer`` must preserve the
      ``__wrapped__`` chain (i.e. use ``functools.wraps``); otherwise the policy
      is attached to a different object than ``@infer`` reads, and is silently
      ignored.
    - The policy is recorded on the function object itself, so the same function
      object cannot be reused to build variants with different policies — every
      decoration shares one policy list.
    """

    if not isinstance(p, Policy):
        raise TypeError(
            "@policy must be called with a Policy instance, "
            "e.g. @policy(CustomPolicy(middleware=lambda: [Retrier(max_retries=5)]))."
        )

    def decorator(func: C) -> C:
        # Attach metadata to the innermost function so it lives in one place
        # regardless of decorator order or intermediate wrappers.
        metadata = _metadata.ensure_metadata(func)
        metadata.setdefault(_metadata.POLICIES_KEY, []).append(p)
        return func

    return decorator


def _partition_middleware(
    middleware: list,
) -> tuple[list[InferenceMiddleware], list[StepMiddleware]]:
    """Split policy-supplied middleware into the inference and step seams."""
    inference_middlewares: list[InferenceMiddleware] = []
    step_middlewares: list[StepMiddleware] = []
    for m in middleware:
        if isinstance(m, InferenceMiddleware):
            inference_middlewares.append(m)
        elif isinstance(m, StepMiddleware):
            step_middlewares.append(m)
        else:
            raise TypeError(
                "Policy middleware must be an instance of InferenceMiddleware "
                f"or StepMiddleware, got {type(m).__name__}"
            )
    return inference_middlewares, step_middlewares


def infer(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator that enables a function's implementation to be inferred by an LLM.
    The function body is ignored; its signature and docstring are used as a prompt.

    A per-function policy can be attached with the ``@policy`` decorator, which
    stores it under the ``"policies"`` key of ``__sefia_metadata__``. Any
    decorator stacked between ``@infer`` and ``@policy`` must preserve the
    ``__wrapped__`` chain (``functools.wraps``) for the policy to be found.
    """

    # The decorator hierarchy is static after decoration, so resolve the
    # innermost function once here rather than on every invocation.
    unwrapped = inspect.unwrap(func)

    @functools.wraps(func)
    async def _run(*args, **kwargs):
        context = get_context()
        # Read policy metadata from the innermost function so the decorator
        # order does not matter and intermediate wrappers are handled.
        metadata = _metadata.get_metadata(unwrapped)
        fn_policies = metadata.get(_metadata.POLICIES_KEY, [])
        all_policies = list(context.policies) + fn_policies
        all_handlers = [
            handler for p in all_policies for handler in p.create_handlers()
        ]
        all_middleware = [
            middleware for p in all_policies for middleware in p.create_middleware()
        ]
        publisher = EventPublisher(all_handlers)
        inference_middlewares, step_middlewares = _partition_middleware(all_middleware)

        executor = InferenceExecutor(
            func=unwrapped,
            args=args,
            kwargs=kwargs,
            inference_strategy=context.inference_strategy,
            tool_collector=context.tool_collector,
            engrave=engrave,
            publisher=publisher,
            inference_middlewares=inference_middlewares,
            step_middlewares=step_middlewares,
        )

        # Only the inference itself is engraved, so glyff can replay it. The
        # setup above (resolving policies and partitioning middleware) runs
        # outside the engrave boundary, so a misconfigured policy surfaces as
        # an ordinary error instead of an engraved, replay-forever failure.
        # The engraved call takes the user's args so glyff keys the record on
        # them; the executor is captured by closure.
        @engrave
        @functools.wraps(func)
        async def _engraved_run(*_args, **_kwargs):
            return await executor.run()

        return await _engraved_run(*args, **kwargs)

    return cast(Callable[P, R], _run)
