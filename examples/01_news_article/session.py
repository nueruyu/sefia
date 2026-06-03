from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Represents the generic state of our long-running application."""

    initial_topic: str
    user_inputs: dict[str, str] = field(default_factory=dict)
    pending_interaction_id: str | None = None
