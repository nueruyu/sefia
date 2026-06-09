import functools
import inspect
from typing import Callable

from glyff import engrave

from .context import get_context
from .event_publisher import EventPublisher
from .executor import InferenceExecutor
from .interfaces import Policy

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


def policy(policy: Policy) -> Callable:
    """
    Decorator that attaches an inference policy to an ``@infer`` function.

    The policy is recorded under the ``"policies"`` key of the function's
    ``__sefia_metadata__`` dict, where ``@infer`` reads it. The order relative
    to ``@infer`` does not matter::

        @infer
        @policy(MaxRetries(count=5))
        async def step(...): ...

    To apply more than one policy, merge them on the caller side (or stack
    multiple ``@policy`` decorators).
    """

    def decorator(func: Callable) -> Callable:
        # Attach metadata to the innermost function so it lives in one place
        # regardless of decorator order or intermediate wrappers.
        underlying = inspect.unwrap(func)
        metadata = getattr(underlying, METADATA_ATTR, None)
        if metadata is None:
            metadata = {}
            setattr(underlying, METADATA_ATTR, metadata)
        metadata.setdefault(POLICIES_KEY, []).append(policy)
        return func

    return decorator


def infer(func: Callable) -> Callable:
    """
    Decorator that enables a function's implementation to be inferred by an LLM.
    The function body is ignored; its signature and docstring are used as a prompt.

    A per-function policy can be attached with the ``@policy`` decorator, which
    stores it under the ``"policies"`` key of ``__sefia_metadata__``.
    """

    # The decorator hierarchy is static after decoration, so resolve the
    # innermost function once here rather than on every invocation.
    unwrapped = inspect.unwrap(func)

    @engrave
    @functools.wraps(func)
    async def _run(*args, **kwargs):
        context = get_context()
        # Read policy metadata from the innermost function so the decorator
        # order does not matter and intermediate wrappers are handled.
        metadata = getattr(unwrapped, METADATA_ATTR, {})
        fn_policies = metadata.get(POLICIES_KEY, [])
        all_policies = list(context.policies) + list(fn_policies)
        all_handlers = [
            handler
            for p in all_policies
            for handler in p.create_handlers()
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

    return _run
