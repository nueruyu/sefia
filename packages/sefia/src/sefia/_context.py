import contextvars
from collections.abc import Hashable
from dataclasses import dataclass, field

from glyff import Session as GlyffSession

from ._interfaces import InferenceStrategy, Policy
from ._interfaces.history_storage import HistoryStorage
from ._tool_system import ToolCollector


@dataclass(frozen=True)
class ProfileBinding:
    """A registered profile's inference strategy and the policies it contributes."""

    strategy: InferenceStrategy
    policies: tuple[Policy, ...]


@dataclass(frozen=True)
class SessionContext:
    """Holds the context for an ongoing sefia inference session."""

    glyff_session: GlyffSession
    inference_strategy: InferenceStrategy
    policies: tuple[Policy, ...]
    tool_collector: ToolCollector
    history_storage: HistoryStorage
    _profiles: dict[Hashable, ProfileBinding] = field(
        default_factory=dict[Hashable, ProfileBinding]
    )

    def resolve_profile(
        self, profile_key: Hashable | None
    ) -> tuple[InferenceStrategy, tuple[Policy, ...]]:
        """
        Resolve a selected profile to its strategy and policies.

        ``None`` (no ``@profile``) yields the session default and no extra
        policies; an unknown key raises with the registered keys listed.
        """
        if profile_key is None:
            return self.inference_strategy, ()
        try:
            binding = self._profiles[profile_key]
        except KeyError:
            available = ", ".join(repr(k) for k in self._profiles) or "(none)"
            raise RuntimeError(
                f"Unknown profile {profile_key!r}. "
                f"Registered profiles: {available}. "
                "Add it to the Session via profiles=[Profile(...)]."
            ) from None
        return binding.strategy, binding.policies


context_var = contextvars.ContextVar[SessionContext]("sefia_context")


def get_context() -> SessionContext:
    """Retrieves the current inference context."""
    try:
        return context_var.get()
    except LookupError:
        raise RuntimeError(
            "Inference context is not set. Are you running outside a sefia.Session?"
        )
