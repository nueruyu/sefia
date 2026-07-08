from dataclasses import dataclass


class UnknownSessionError(Exception):
    """Raised when a requested HTTP session is not known."""

    def __init__(self, session_id: str):
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = session_id


class UnknownHumanInputError(Exception):
    """Raised when an input targets an unknown pending human input."""

    def __init__(self, interaction_id: str):
        super().__init__(f"Unknown pending human input: {interaction_id}")
        self.interaction_id = interaction_id


class AmbiguousHumanInputError(Exception):
    """Raised when multiple pending human inputs need an explicit reply target."""

    def __init__(self, interaction_ids: list[str]):
        super().__init__(
            "Multiple pending human inputs exist. Specify one with reply_to: "
            + ", ".join(interaction_ids)
        )
        self.interaction_ids = interaction_ids


@dataclass(frozen=True)
class InputRequired(Exception):
    """Raised when a session pauses to wait for external input."""

    interaction_id: str
    question: str

    def __str__(self) -> str:
        return f"Input required: {self.question}"
