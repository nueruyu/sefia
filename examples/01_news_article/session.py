from pydantic import BaseModel, Field


class SessionState(BaseModel):
    """Represents the generic state of our long-running application."""

    initial_topic: str | None = None
    user_inputs: dict[str, str] = Field(default_factory=dict)
    pending_interaction_id: str | None = None
