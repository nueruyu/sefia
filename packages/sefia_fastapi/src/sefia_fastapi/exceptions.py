class UnknownSessionError(Exception):
    """Raised when a requested HTTP session is not known."""

    def __init__(self, session_id: str):
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = session_id


class UnknownInputError(Exception):
    """Raised when an input targets an unknown pending input."""

    def __init__(self, interaction_id: str):
        super().__init__(f"Unknown pending input: {interaction_id}")
        self.interaction_id = interaction_id


class AmbiguousInputError(Exception):
    """Raised when multiple pending inputs need an explicit reply target."""

    def __init__(self, interaction_ids: list[str]):
        super().__init__(
            "Multiple pending inputs exist. Specify one with reply_to: "
            + ", ".join(interaction_ids)
        )
        self.interaction_ids = interaction_ids
