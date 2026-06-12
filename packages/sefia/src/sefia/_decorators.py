import functools
import inspect
from typing import Callable, TypeVar

from glyff import engrave

from ._context import get_context
from ._executor import InferenceExecutor
from ._interfaces import InferenceMiddleware, Policy, StepMiddleware
from .event_system import EventPublisher


T = TypeVar("T")

METADATA_ATTR = "__sefia_metadata__"
POLICIES_KEY = "policies"


def get_metadata(func: Callable) -> dict:
    return getattr(inspect.unwrap(func), METADATA_ATTR, {})


def tool(func: T) -> T:
    target = func.__func__ if isinstance(func, (classmethod, staticmethod)) else func
    setattr(target, "__sefia_tool__", True)
    return func


def policy(policy: Policy) -> Callable:
    if not isinstance(policy, Policy):
        raise TypeError(
            "@policy must be called with a Policy instance, "
            "e.g. @policy(MaxRetries(count=5))."
        )

    def decorator(func: Callable) -> Callable:
        underlying = inspect.unwrap(func)
        metadata = getattr(underlying, METADATA_ATTR, None)
        if metadata is None:
            metadata = {}
            setattr(underlying, METADATA_ATTR, metadata)
        metadata.setdefault(POLICIES_KEY, []).append(policy)
        return func

    return decorator


def _partition_middleware(
    middleware: list,
) -> tuple[list[InferenceMiddleware], list[StepMiddleware]]:
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


def infer(func: Callable) -> Callable:
    unwrapped = inspect.unwrap(func)

    @functools.wraps(func)
    async def _run(*args, **kwargs):
        context = get_context()
        metadata = getattr(unwrapped, METADATA_ATTR, {})
        fn_policies = metadata.get(POLICIES_KEY, [])
        all_policies = list(context.policies) + fn_policies
        all_handlers = [
            handler
            for p in all_policies
            for handler in p.create_handlers()
        ]
        all_middleware = [
            middleware
            for p in all_policies
            for middleware in p.create_middleware()
        ]
        publisher = EventPublisher(all_handlers)
        inference_middlewares, step_middlewares = _partition_middleware(
            all_middleware
        )

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

        @engrave
        @functools.wraps(func)
        async def _engraved_run(*_args, **_kwargs):
            return await executor.run()

        return await _engraved_run(*args, **kwargs)

    return _run
