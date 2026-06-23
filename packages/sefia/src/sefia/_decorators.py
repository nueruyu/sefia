import functools
import inspect
from typing import Callable, ParamSpec, Protocol, TypeVar, cast

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


class _PolicyDecorator(Protocol):
    """Callable that decorates a function without changing its type."""

    def __call__(self, func: C) -> C: ...


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

    def decorator(func: C) -> C:
        # Attach metadata to the innermost function so it lives in one place
        # regardless of decorator order or intermediate wrappers.
        metadata = _metadata.ensure_metadata(func)
        metadata.setdefault(_metadata.POLICIES_KEY, []).append(p)
        return func

    return decorator


def model(profile_name: str) -> _PolicyDecorator:
    """
    Decorator that selects the model profile an ``@infer`` function runs on.

    Instead of writing a raw model name at the call site, you reference a
    :class:`~sefia.ModelProfile` by name. Profiles are built up front and
    registered on the :class:`~sefia.Session` (``profiles=[...]``); this
    decorator only records which one to use, so the same code can bind to a
    different concrete client per session (e.g. a mock in tests)::

        @infer
        @model("fast")
        async def step(...): ...

    The name is recorded under the ``"model_profile"`` key of the function's
    ``__sefia_metadata__`` dict, where ``@infer`` reads it. Like ``@policy``, the
    order relative to ``@infer`` does not matter, and any decorator stacked
    between the two must preserve the ``__wrapped__`` chain (``functools.wraps``)
    for the selection to be found. A function has a single model profile; a later
    ``@model`` decoration overrides an earlier one. When no ``@model`` is present,
    the session's default ``llm_client`` is used.

    The referenced profile must exist on the session; an unknown name raises at
    call time with the list of registered profiles.
    """

    if not isinstance(profile_name, str):
        raise TypeError(
            "@model must be called with a profile name (str), "
            'e.g. @model("fast").'
        )

    def decorator(func: C) -> C:
        # Attach metadata to the innermost function so it lives in one place
        # regardless of decorator order or intermediate wrappers.
        metadata = _metadata.ensure_metadata(func)
        metadata[_metadata.MODEL_PROFILE_KEY] = profile_name
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

        # Resolve the model profile selected by @model, falling back to the
        # session default when none was attached.
        profile_name = metadata.get(_metadata.MODEL_PROFILE_KEY)
        inference_strategy = context.resolve_inference_strategy(profile_name)

        executor = InferenceExecutor(
            func=unwrapped,
            args=args,
            kwargs=kwargs,
            inference_strategy=inference_strategy,
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
