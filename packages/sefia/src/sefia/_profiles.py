from collections.abc import Sequence
from dataclasses import dataclass, field

from ._interfaces import Policy
from .llm._client import LLMClient


@dataclass(frozen=True)
class ModelProfile:
    """
    A named, reusable bundle of inference configuration for ``@infer`` functions.

    A profile pairs a ``name`` with the :class:`~sefia.llm.LLMClient` used to run
    inference, plus any :class:`~sefia.Policy` objects that should apply to every
    function that selects it. Profiles are registered up front on the
    :class:`~sefia.Session` (``profiles=[...]``) and selected per function by name
    with ``@profile("<name>")``, so the selection at the call site is decoupled
    from the concrete client — a test can bind the same name to a mock.

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

    name: str
    client: LLMClient
    policies: Sequence[Policy] = field(default=())

    def __post_init__(self) -> None:
        # Accept any sequence of policies (e.g. a list at the call site, like
        # Session(policies=[...])) but store an immutable tuple so the profile
        # never holds a shared, mutable reference.
        object.__setattr__(self, "policies", tuple(self.policies))
