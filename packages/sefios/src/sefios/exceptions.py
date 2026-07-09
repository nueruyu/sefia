from sefia.exceptions import PauseException


class NeedsInput(PauseException):
    """
    Raised by an input-awaiting tool to pause the run until input is available.

    Carries the ``prompt`` shown to whoever provides the input. Catch it to
    surface the pause to your caller; once the input is recorded, re-invoking
    the same session replays the completed steps and re-runs the tool, which
    now returns the input.

    It subclasses :class:`sefia.exceptions.PauseException`, so the sefia executor
    propagates it as a pause (never reporting it as a failure) without the core
    needing to know about external input specifically.
    """

    def __init__(self, prompt: str) -> None:
        super().__init__(prompt)
        self.prompt = prompt
