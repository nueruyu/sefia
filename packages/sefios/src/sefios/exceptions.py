from sefia.exceptions import PauseException


class UnknownInputError(Exception):
    """Raised when input targets an unknown pending request."""

    def __init__(self, interaction_id: str):
        super().__init__(f"Unknown pending input: {interaction_id}")
        self.interaction_id = interaction_id


class AmbiguousInputError(Exception):
    """Raised when input cannot be routed among multiple pending requests."""

    def __init__(self, interaction_ids: list[str]):
        super().__init__(
            "Multiple pending inputs exist. Specify one with reply_to: "
            + ", ".join(interaction_ids)
        )
        self.interaction_ids = interaction_ids


class InputRequired(PauseException):
    """
    Raised by an input-awaiting tool to pause the run until input is available.
    """

    def __init__(self, prompt: str, *, interaction_id: str | None = None) -> None:
        super().__init__(prompt)
        self.prompt = prompt
        self.interaction_id = interaction_id


class UnknownExecutionError(Exception):
    """Raised when an opaque durable execution reference is unknown."""

    def __init__(self, execution_id: str):
        super().__init__(f"Unknown execution: {execution_id}")
        self.execution_id = execution_id


class ExecutionConflictError(Exception):
    """Raised when immutable execution data conflicts with an existing value."""

    def __init__(self, execution_id: str):
        super().__init__(f"Execution data conflicts for {execution_id}")
        self.execution_id = execution_id


class ExecutionAlreadyTerminalError(Exception):
    """Raised when a terminal operation conflicts with an existing terminal state."""

    def __init__(self, execution_id: str):
        super().__init__(f"Execution is already terminal: {execution_id}")
        self.execution_id = execution_id


class UnknownExternalActionError(Exception):
    """Raised when a result targets an action not requested by the execution."""

    def __init__(self, action_id: str):
        super().__init__(f"Unknown external action: {action_id}")
        self.action_id = action_id


class ExternalActionConflictError(Exception):
    """Raised when immutable request/result data conflicts for an action."""

    def __init__(self, action_id: str):
        super().__init__(f"External action data conflicts for {action_id}")
        self.action_id = action_id
