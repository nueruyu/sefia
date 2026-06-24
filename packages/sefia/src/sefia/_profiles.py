from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from ._interfaces import Policy
from .llm._client import LLMClient


@dataclass(frozen=True)
class Profile:
    """
    A keyed, reusable bundle of inference configuration for ``@infer`` functions.

    A profile pairs a ``key`` with the :class:`~sefia.llm.LLMClient` used to run
    inference, plus any :class:`~sefia.Policy` objects that should apply to every
    function that selects it. Profiles are registered up front on the
    :class:`~sefia.Session` (``profiles=[...]``) and selected per function by key
    with ``@profile(<key>)``, so the selection at the call site is decoupled from
    the concrete client — a test can bind the same key to a mock.

    The ``key`` is any hashable value, not just a string, so an application can
    use an ``Enum`` member (or any other hashable) to avoid stringly-typed
    configuration::

        class Models(Enum):
            FAST = auto()
            SMART = auto()

        Session(profiles=[Profile(key=Models.SMART, client=...)])

        @infer
        @profile(Models.SMART)
        async def step(...): ...

    Configuration is layered, most specific wins:

        function-explicit (``@policy`` / ``@profile``)  >  profile  >  session

    The selected profile's ``client`` overrides the session's default
    ``llm_client``, and its ``policies`` are applied on top of the session
    policies and beneath the function's own ``@policy`` decorators.

    Model *settings* (temperature, max tokens, ...) are captured today by how the
    client is constructed (e.g. ``LiteLLMClient(model=..., temperature=0.2)``).
    The profile is the seam where first-class, client-agnostic settings can be
    added later without touching any call site.
    """

    key: Hashable
    client: LLMClient
    policies: Sequence[Policy] = ()

    def __post_init__(self) -> None:
        # The key indexes the session's profile registry (a dict), so it must be
        # hashable and not the None sentinel that marks "no @profile". Validate
        # up front with a clear message instead of a later, opaque TypeError.
        if self.key is None:
            raise TypeError("Profile key must not be None.")
        try:
            hash(self.key)
        except TypeError as e:
            raise TypeError(
                f"Profile key must be hashable, got {type(self.key).__name__}."
            ) from e
