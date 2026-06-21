import functools
import inspect
from typing import Any, Callable, Coroutine, ParamSpec, Protocol, TypeVar

from glyff import engrave

from ._context import get_context
from ._executor import InferenceExecutor
from ._interfaces import InferenceMiddleware, Policy, StepMiddleware
from .event_system import EventPublisher

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


class _PolicyDecorator(Protocol):
    def __call__(self, func: T) -> T: ...

# Attribute that holds sefia's per-function metadata dict, and the key under
# which inference policies live inside it.
METADATA_ATTR = "__sefia_metadata__"
POLICIES_KEY = "policies"


def get_metadata(func: Callable) -> dict:
    """
    Return the sefia metadata dict attached to ``func`` (empty if there is none).

    The lookup unwraps ``functools.wraps`` layers, so it works on a function
    regardless of which decorators wrap it.
    """
    return getattr(inspect.unwrap(func), METADATA_ATTR, {})


def tool(func: T) -> T:
    """
    Mark a method as a tool available to an @infer step.

    This is a pure marker: the inference executor normalizes sync/async return
    values, so the wrapped function is returned unchanged to preserve its
    signature for schema generation.
    """
    # When @tool is applied over @classmethod/@staticmethod, mark the underlying
    # function — those descriptor objects may reject attribute assignment.
    target = func.__func__ if isinstance(func, (classmethod, staticmethod)) else func
    setattr(target, "__sefia_tool__", True)
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

    def decorator(func: T) -> T:
        # Attach metadata to the innermost function so it lives in one place
        # regardless of decorator order or intermediate wrappers.
        underlying = inspect.unwrap(func)  # type: ignore[arg-type]
        metadata = getattr(underlying, METADATA_ATTR, None)
        if metadata is None:
            metadata = {}
            setattr(underlying, METADATA_ATTR, metadata)
        metadata.setdefault(POLICIES_KEY, []).append(p)
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
        metadata = getattr(unwrapped, METADATA_ATTR, {})
        fn_policies = metadata.get(POLICIES_KEY, [])
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

    return _run  # type: ignore[return-value]
