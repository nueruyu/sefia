from sefia.exceptions import PauseException


class NeedsInput(PauseException):
    """
    Raised by an input-awaiting tool (such as human input) to pause the run
    until an answer is available.

    Carries the ``question`` that needs answering. Catch it to surface the pause
    to your caller; once the answer is recorded, re-invoking the same session
    replays the completed steps and re-runs the tool, which now returns the
    answer.

    It subclasses :class:`sefia.exceptions.PauseException`, so the sefia executor
    propagates it as a pause (never reporting it as a failure) without the core
    needing to know about human input specifically.
    """

    def __init__(self, question: str) -> None:
        super().__init__(question)
        self.question = question
