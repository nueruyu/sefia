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

    The policies are stored under the ``"policies"`` key of the function's
    ``__sefia_metadata__`` dict, where ``@infer`` reads them. Apply it below
    ``@infer`` (closer to the function)::

        @infer
        @with_policies([MaxRetries(count=5)])
        async def step(...): ...
    """

    def decorator(func: Callable) -> Callable:
        metadata = getattr(func, "__sefia_metadata__", None)
        if metadata is None:
            metadata = {}
            setattr(func, "__sefia_metadata__", metadata)
        metadata["policies"] = list(policies)
        return func

    return decorator


def infer(func: Callable) -> Callable:
    """
    Decorator that enables a function's implementation to be inferred by an LLM.
    The function body is ignored; its signature and docstring are used as a prompt.

    Per-function policies can be attached with the ``@policies`` decorator, which
    stores them under the ``"policies"`` key of ``__sefia_metadata__``.
    """

    metadata = getattr(func, "__sefia_metadata__", {})
    fn_policies = metadata.get("policies", [])

    @engrave
    @functools.wraps(func)
    async def _run(*args, **kwargs):
        context = get_context()
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
