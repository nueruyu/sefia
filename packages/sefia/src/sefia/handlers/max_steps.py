from typing import Type

from sefia.events import Event, StepStarted
from sefia.interfaces import EventHandler


class MaxStepsExceededError(Exception):
    """Raised when the inference loop exceeds its maximum number of steps."""

    pass


class MaxStepsHandler(EventHandler[StepStarted]):
    """
    Stops the inference loop once it exceeds a maximum number of steps.

    The executor owns the step counter and reports it via ``StepStarted`` as a
    0-based index; this handler raises ``MaxStepsExceededError`` once that index
    reaches ``max_steps`` (i.e. after ``max_steps`` steps have run). Pass
    ``None`` for no limit.
    """

    def __init__(self, max_steps: int | None):
        self.max_steps = max_steps

    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        return (StepStarted,)

    async def handle(self, event: StepStarted) -> None:
        if self.max_steps is not None and event.step >= self.max_steps:
            raise MaxStepsExceededError(
                f"Inference exceeded the maximum of {self.max_steps} step(s)."
            )
