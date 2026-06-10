import functools
from typing import Callable, TypeVar

from glyff import engrave

from .context import get_context
from .event_publisher import EventPublisher
from .executor import InferenceExecutor
from .interfaces import Policy

T = TypeVar("T")


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


def infer(policies: list[Policy] | None = None) -> Callable:
    """
    Decorator that enables a function's implementation to be inferred by an LLM.
    The function body is ignored; its signature and docstring are used as a prompt.
    """

    def decorator(func: Callable) -> Callable:
        @engrave
        @functools.wraps(func)
        async def _run(*args, **kwargs):
            context = get_context()
            all_policies = context.policies + (policies or [])
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

    return decorator
