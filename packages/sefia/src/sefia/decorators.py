import functools
import inspect
from typing import Callable

from glyff import engrave

from .context import get_context
from .event_publisher import EventPublisher
from .executor import InferenceExecutor
from .interfaces import InferenceMiddleware, Policy, StepMiddleware


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
            all_middleware = [
                middleware
                for policy in all_policies
                for middleware in policy.create_middleware()
            ]
            publisher = EventPublisher(all_handlers)

            inference_middlewares: list[InferenceMiddleware] = []
            step_middlewares: list[StepMiddleware] = []
            for m in all_middleware:
                if isinstance(m, InferenceMiddleware):
                    inference_middlewares.append(m)
                elif isinstance(m, StepMiddleware):
                    step_middlewares.append(m)
                else:
                    raise TypeError(
                        "Policy middleware must be an instance of "
                        "InferenceMiddleware or StepMiddleware, got "
                        f"{type(m).__name__}"
                    )

            executor = InferenceExecutor(
                func=func,
                args=args,
                kwargs=kwargs,
                inference_strategy=context.inference_strategy,
                tool_collector=context.tool_collector,
                engrave=engrave,
                publisher=publisher,
                inference_middlewares=inference_middlewares,
                step_middlewares=step_middlewares,
            )
            return await executor.run()

        setattr(_run, "__sefia_infer__", True)
        return _run

    return decorator
