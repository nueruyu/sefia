from __future__ import annotations

import functools
import inspect
from collections.abc import Hashable
from typing import Callable, ParamSpec, TypeVar, cast

import glyff

from .._context import get_context
from .._executor import InferenceExecutor
from .._glyff import engrave
from .._interfaces import InferenceMiddleware, Policy, StepMiddleware
from ..event_system import EventPublisher
from . import metadata

P = ParamSpec("P")
R = TypeVar("R")


def partition_middleware(
    middleware: list,
) -> tuple[list[InferenceMiddleware], list[StepMiddleware]]:
    """Split policy-supplied middleware into the inference and step seams."""
    inference_middlewares: list[InferenceMiddleware] = []
    step_middlewares: list[StepMiddleware] = []
    for item in middleware:
        if isinstance(item, InferenceMiddleware):
            inference_middlewares.append(item)
        elif isinstance(item, StepMiddleware):
            step_middlewares.append(item)
        else:
            raise TypeError(
                "Policy middleware must be an instance of InferenceMiddleware "
                f"or StepMiddleware, got {type(item).__name__}"
            )
    return inference_middlewares, step_middlewares


def decorate_inference(
    func: Callable[P, R],
    *,
    domain: glyff.Domain,
    name: str,
    domain_profile: Hashable | None,
    domain_policies: tuple[Policy, ...],
) -> Callable[P, R]:
    """Build an inference decorator bound to an owning domain and defaults."""
    unwrapped = inspect.unwrap(func)

    @functools.wraps(func)
    async def run(*args, **kwargs):
        context = get_context()
        function_metadata = metadata.get_metadata(unwrapped)
        function_policies = function_metadata.get(metadata.KEY_POLICIES, [])
        profile_key = function_metadata.get(metadata.KEY_PROFILE_KEY, domain_profile)
        inference_strategy, profile_policies = context.resolve_profile(profile_key)

        policies = [
            *context.policies,
            *domain_policies,
            *profile_policies,
            *function_policies,
        ]
        handlers = [
            handler for policy in policies for handler in policy.create_handlers()
        ]
        middleware = [
            item for policy in policies for item in policy.create_middleware()
        ]
        inference_middleware, step_middleware = partition_middleware(middleware)
        executor = InferenceExecutor(
            func=unwrapped,
            args=args,
            kwargs=kwargs,
            inference_strategy=inference_strategy,
            tool_collector=context.tool_collector,
            engrave=None,
            publisher=EventPublisher(handlers),
            inference_middlewares=inference_middleware,
            step_middlewares=step_middleware,
            history_storage=context.history_storage,
        )

        @functools.wraps(func)
        async def engraved_run(*_args, **_kwargs):
            return await executor.run()

        return await engrave(domain, name, engraved_run)(*args, **kwargs)

    return cast(Callable[P, R], run)
