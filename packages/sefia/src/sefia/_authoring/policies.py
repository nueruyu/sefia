from collections.abc import Hashable
from typing import Callable, Protocol, TypeVar

from .._interfaces import Policy
from .._profiles import Profile
from . import metadata

C = TypeVar("C", bound=Callable[..., object])


class PolicyDecorator(Protocol):
    def __call__(self, func: C) -> C: ...


def policy(value: Policy) -> PolicyDecorator:
    """Attach an inference policy to a function."""
    if not isinstance(value, Policy):
        raise TypeError(
            "@policy must be called with a Policy instance, "
            "e.g. @policy(Policy(middleware=lambda: [Retrier(max_retries=5)]))."
        )

    def decorator(func: C) -> C:
        function_metadata = metadata.ensure_metadata(func)
        function_metadata.setdefault(metadata.KEY_POLICIES, []).append(value)
        return func

    return decorator


def profile(profile_key: Hashable) -> PolicyDecorator:
    """Select the profile an inferred function runs on."""
    if profile_key is None:
        raise TypeError("@profile key must not be None.")
    if isinstance(profile_key, Profile):
        raise TypeError(
            "@profile takes the profile's key (e.g. a str or Enum member), "
            "not the Profile instance itself."
        )
    try:
        hash(profile_key)
    except TypeError as error:
        raise TypeError(
            f"@profile key must be hashable, got {type(profile_key).__name__}."
        ) from error

    def decorator(func: C) -> C:
        function_metadata = metadata.ensure_metadata(func)
        function_metadata[metadata.KEY_PROFILE_KEY] = profile_key
        return func

    return decorator
