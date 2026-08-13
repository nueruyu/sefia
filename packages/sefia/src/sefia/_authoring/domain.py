import functools
import inspect
from collections.abc import Hashable, Sequence
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar, cast, overload

import glyff
from typing_extensions import final

from .._context import get_context
from .._executor import InferenceExecutor
from .._glyff import engrave as engrave_call
from .._interfaces import InferenceMiddleware, Policy, StepMiddleware
from ..event_system import EventPublisher
from . import metadata

P = ParamSpec("P")
R = TypeVar("R")


def _partition_middleware(
    middleware: list,
) -> tuple[list[InferenceMiddleware], list[StepMiddleware]]:
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


@final
class Domain:
    """Sefia authoring defaults bound to a versioned Glyff ownership domain."""

    def __init__(
        self,
        domain: glyff.Domain,
        *,
        default_profile: Hashable | None = None,
        policies: Sequence[Policy] = (),
    ) -> None:
        self.glyff = domain
        self.default_profile = default_profile
        self.policies = tuple(policies)

    @overload
    def infer(self, func: Callable[P, R]) -> Callable[P, R]: ...

    @overload
    def infer(self, *, name: str) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def infer(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
    ) -> Any:
        """Decorate an inferred function using its qualified or explicit name."""

        def decorator(func: Callable[P, R], execution_name: str) -> Callable[P, R]:
            return self._decorate_inference(func, execution_name)

        if func is not None:
            return decorator(func, func.__qualname__)
        if not name:
            raise ValueError("An inference execution name cannot be empty.")
        return lambda func: decorator(func, name)

    def _decorate_inference(
        self, func: Callable[P, R], execution_name: str
    ) -> Callable[P, R]:
        unwrapped = inspect.unwrap(func)

        @functools.wraps(func)
        async def run(*args, **kwargs):
            context = get_context()
            function_metadata = metadata.get_metadata(unwrapped)
            function_policies = function_metadata.get(metadata.KEY_POLICIES, [])
            profile_key = function_metadata.get(
                metadata.KEY_PROFILE_KEY, self.default_profile
            )
            inference_strategy, profile_policies = context.resolve_profile(profile_key)

            policies = [
                *context.policies,
                *self.policies,
                *profile_policies,
                *function_policies,
            ]
            handlers = [
                handler for policy in policies for handler in policy.create_handlers()
            ]
            middleware = [
                item for policy in policies for item in policy.create_middleware()
            ]
            inference_middleware, step_middleware = _partition_middleware(middleware)
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

            return await engrave_call(self.glyff, execution_name, engraved_run)(
                *args, **kwargs
            )

        return cast(Callable[P, R], run)

    @overload
    def engrave(self, func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]: ...

    @overload
    def engrave(
        self, *, name: str
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

    def engrave(
        self,
        func: Callable[..., Awaitable[Any]] | None = None,
        *,
        name: str | None = None,
    ) -> Any:
        """Decorate a function using its name or an explicit stable name."""
        if func is not None:
            return engrave_call(self.glyff, func.__name__, func)
        if not name:
            raise ValueError("An execution name cannot be empty.")

        def decorator(func):
            return engrave_call(self.glyff, name, func)

        return decorator
