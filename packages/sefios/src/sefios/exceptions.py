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

    Carries the ``prompt`` shown to whoever provides the input, and the
    ``interaction_id`` identifying the paused request so integration layers
    can report exactly which request is waiting without re-reading state.
    Catch it to surface the pause to your caller; once the input is recorded,
    re-invoking the same session replays the completed steps and re-runs the
    tool, which now returns the input.

    It subclasses :class:`sefia.exceptions.PauseException`, so the sefia executor
    propagates it as a pause (never reporting it as a failure) without the core
    needing to know about external input specifically.
    """

    def __init__(self, prompt: str, *, interaction_id: str | None = None) -> None:
        super().__init__(prompt)
        self.prompt = prompt
        self.interaction_id = interaction_id
