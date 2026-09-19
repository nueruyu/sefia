from pydantic import JsonValue
from sefia.exceptions import PauseException


class UnknownInteractionError(Exception):
    """Resolution targeted an interaction with no durable request."""

    def __init__(self, interaction_id: str):
        super().__init__(f"Unknown interaction: {interaction_id}")
        self.interaction_id = interaction_id


class InteractionConflictError(Exception):
    """A request or result contradicts a previously persisted fact."""

    def __init__(self, interaction_id: str):
        super().__init__(f"Conflicting interaction: {interaction_id}")
        self.interaction_id = interaction_id


class InteractionRequired(PauseException):
    """Pause until an external consumer resolves the durable request."""

    def __init__(self, interaction_id: str, request: JsonValue):
        super().__init__(f"Interaction required: {interaction_id}")
        self.interaction_id = interaction_id
        self.request = request
