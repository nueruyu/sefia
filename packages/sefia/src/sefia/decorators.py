import functools
import inspect
from typing import Callable

from glyff import engrave

from .context import get_context
from .event_publisher import EventPublisher
from .executor import InferenceExecutor
from .interfaces import Policy


def tool(func: Callable) -> Callable:
    """
    Decorator to mark a method as a tool available to the LLM.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    setattr(wrapper, "__sefia_tool__", True)
    return wrapper


def with_policies(policies: list[Policy]) -> Callable:
    """
    Decorator that attaches inference policies to an ``@infer`` function.

    The policies are appended under the ``"policies"`` key of the function's
    ``__sefia_metadata__`` dict, where ``@infer`` reads them. The order relative
    to ``@infer`` does not matter, and stacking multiple ``@with_policies``
    decorators accumulates their policies rather than overwriting::

        @infer
        @with_policies([MaxRetries(count=5)])
        async def step(...): ...
    """

    def decorator(func: Callable) -> Callable:
        # Attach metadata to the innermost function so it lives in one place
        # regardless of decorator order or intermediate wrappers.
        underlying = inspect.unwrap(func)
        metadata = getattr(underlying, "__sefia_metadata__", None)
        if metadata is None:
            metadata = {}
            setattr(underlying, "__sefia_metadata__", metadata)
        metadata.setdefault("policies", []).extend(policies)
        # Mirror it onto the outer object too, so the metadata is visible whether
        # @with_policies sits above or below @infer (and to plain introspection).
        if func is not underlying:
            setattr(func, "__sefia_metadata__", metadata)
        return func

    return decorator


def infer(func: Callable) -> Callable:
    """
    Decorator that enables a function's implementation to be inferred by an LLM.
    The function body is ignored; its signature and docstring are used as a prompt.

    Per-function policies can be attached with the ``@with_policies`` decorator,
    which stores them under the ``"policies"`` key of ``__sefia_metadata__``.
    """

    @engrave
    @functools.wraps(func)
    async def _run(*args, **kwargs):
        context = get_context()
        # Read policy metadata from the innermost function at runtime, so the
        # decorator order does not matter and intermediate wrappers are handled.
        metadata = getattr(inspect.unwrap(func), "__sefia_metadata__", {})
        fn_policies = metadata.get("policies", [])
        all_policies = context.policies + fn_policies
        all_handlers = [
            handler
            for policy in all_policies
            for handler in policy.create_handlers()
        ]
        publisher = EventPublisher(all_handlers)

        executor = InferenceExecutor(
            func=func,
            args=args,
            kwargs=kwargs,
            inference_strategy=context.inference_strategy,
            tool_collector=context.tool_collector,
            engrave=engrave,
            publisher=publisher,
        )
        return await executor.run()

    setattr(_run, "__sefia_infer__", True)
    return _run
