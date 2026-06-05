from typing import Type, Union

from sefia.events import AttemptStart, BeforeInferenceStep, Event
from sefia.interfaces import EventHandler


class MaxTurnsExceededError(Exception):
    """Raised when the inference loop exceeds the maximum number of turns."""

    pass


class MaxTurnsHandler(EventHandler[Union[AttemptStart, BeforeInferenceStep]]):
    """
    Enforces a limit on the number of turns within a single inference attempt.

    Counts BeforeInferenceStep events to track how many turns have elapsed and
    resets the counter on AttemptStart so each retry attempt starts fresh.
    Raises MaxTurnsExceededError when the limit is exceeded.
    """

    def __init__(self, max_turns: int = 25):
        self.max_turns = max_turns
        self._turns_used = 0

    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        return (AttemptStart, BeforeInferenceStep)

    async def handle(
        self, event: Union[AttemptStart, BeforeInferenceStep]
    ) -> None:
        if isinstance(event, AttemptStart):
            self._turns_used = 0
        elif isinstance(event, BeforeInferenceStep):
            self._turns_used += 1
            if self._turns_used > self.max_turns:
                raise MaxTurnsExceededError(
                    f"Inference did not complete within {self.max_turns} turns."
                )
