import functools
import inspect
from collections.abc import Hashable, Sequence
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar, cast, overload

import glyff
from typing_extensions import final

from .._context import get_context
from .._executor import InferenceExecutor
from .._interfaces import InferenceMiddleware, Policy, StepMiddleware
from ..event_system import EventPublisher
from . import metadata

P = ParamSpec("P")
R = TypeVar("R")

GLYFF_DOMAIN = glyff.Domain("sefia", version="1")


def _partition_middleware(
    middleware: Sequence[object],
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

    __slots__ = ("_glyff", "_default_profile", "_policies")

    def __init__(
        self,
        domain: glyff.Domain,
        *,
        default_profile: Hashable | None = None,
        policies: Sequence[Policy] = (),
    ) -> None:
        self._glyff = domain
        self._default_profile = default_profile
        self._policies = tuple(policies)

    @overload
    def infer(self, func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]: ...

    @overload
    def infer(
        self, *, name: str
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...

    def infer(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
    ) -> Any:
        """Decorate an inferred function using its qualified or explicit name."""

        def decorator(
            func: Callable[P, Awaitable[R]], execution_name: str | None
        ) -> Callable[P, Awaitable[R]]:
            return self._decorate_inference(func, execution_name)

        if func is not None:
            return decorator(func, None)
        if not name:
            raise ValueError("An inference execution name cannot be empty.")

        def named_decorator(
            func: Callable[P, Awaitable[R]],
        ) -> Callable[P, Awaitable[R]]:
            return decorator(func, name)

        return named_decorator

    def _decorate_inference(
        self, func: Callable[P, Awaitable[R]], execution_name: str | None
    ) -> Callable[P, Awaitable[R]]:
        unwrapped = inspect.unwrap(func)

        @functools.wraps(func)
        async def run(*args: P.args, **kwargs: P.kwargs) -> R:
            context = get_context()
            function_metadata = metadata.get_metadata(unwrapped)
            function_policies = cast(
                list[Policy], function_metadata.get(metadata.KEY_POLICIES, [])
            )
            profile_key = cast(
                Hashable | None,
                function_metadata.get(metadata.KEY_PROFILE_KEY, self._default_profile),
            )
            inference_strategy, profile_policies = context.resolve_profile(profile_key)

            policies = [
                *context.policies,
                *self._policies,
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
                engrave=lambda name, func: GLYFF_DOMAIN.engrave(func, name=name),
                publisher=EventPublisher(handlers),
                inference_middlewares=inference_middleware,
                step_middlewares=step_middleware,
                history_storage=context.history_storage,
            )

            @functools.wraps(func)
            async def engraved_run(*_args: P.args, **_kwargs: P.kwargs) -> R:
                return await executor.run()

            return await self._glyff.engrave(engraved_run, name=execution_name)(
                *args, **kwargs
            )

        return cast(Callable[P, Awaitable[R]], run)

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
        """Decorate a function using its qualified or explicit name."""
        if func is not None:
            return self._glyff.engrave(func)
        if not name:
            raise ValueError("An execution name cannot be empty.")

        def decorator(
            func: Callable[P, Awaitable[R]],
        ) -> Callable[P, Awaitable[R]]:
            return self._glyff.engrave(func, name=name)

        return decorator
