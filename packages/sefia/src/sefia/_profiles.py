from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from ._interfaces import Policy
from .llm._client import LLMClient


@dataclass(frozen=True)
class Profile:
    """
    A keyed, reusable bundle of inference configuration for ``@infer`` functions.

    A profile pairs a ``key`` with the :class:`~sefia.llm.LLMClient` used to run
    inference, plus any :class:`~sefia.Policy` objects that apply whenever it is
    selected. Profiles are registered on the :class:`~sefia.Session`
    (``profiles=[...]``) and chosen per function with ``@profile(<key>)``. The
    ``key`` is any hashable (a string, an ``Enum`` member, ...), so configuration
    need not be stringly typed. The client overrides the session default, and the
    policies layer between the session and the function's own ``@policy``.
    """

    key: Hashable
    client: LLMClient
    policies: Sequence[Policy] = ()

    def __post_init__(self) -> None:
        # The key indexes the session's profile registry (a dict).
        if self.key is None:
            raise TypeError("Profile key must not be None.")
        try:
            hash(self.key)
        except TypeError as e:
            raise TypeError(
                f"Profile key must be hashable, got {type(self.key).__name__}."
            ) from e
