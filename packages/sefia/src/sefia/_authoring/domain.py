from collections.abc import Hashable, Sequence
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar, overload

import glyff
from typing_extensions import final

from .._glyff import engrave as engrave_call
from .._interfaces import Policy
from .inference import decorate_inference

P = ParamSpec("P")
R = TypeVar("R")


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
            return decorate_inference(
                func,
                domain=self.glyff,
                name=execution_name,
                domain_profile=self.default_profile,
                domain_policies=self.policies,
            )

        if func is not None:
            return decorator(func, func.__qualname__)
        if not name:
            raise ValueError("An inference execution name cannot be empty.")
        return lambda func: decorator(func, name)

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
