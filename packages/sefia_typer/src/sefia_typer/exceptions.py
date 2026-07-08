class UnknownSessionError(Exception):
    """Raised when a requested CLI session is not known."""

    def __init__(self, session_id: str):
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = session_id


class UnknownHumanInputError(Exception):
    """Raised when a CLI input targets an unknown pending human input."""

    def __init__(self, interaction_id: str):
        super().__init__(f"Unknown pending human input: {interaction_id}")
        self.interaction_id = interaction_id


class AmbiguousHumanInputError(Exception):
    """Raised when multiple pending human inputs need an explicit reply target."""

    def __init__(self, interaction_ids: list[str]):
        super().__init__(
            "Multiple pending human inputs exist. Specify one with --reply-to: "
            + ", ".join(interaction_ids)
        )
        self.interaction_ids = interaction_ids
