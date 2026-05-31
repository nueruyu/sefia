from typing import Type, Union

from sefia.events import AttemptStart, Event, InferenceFailed, ToolExecutionFailed
from sefia.interfaces import EventHandler


class RequestInferenceRetry(Exception):
    """Signal to the executor to retry the entire inference process."""

    pass


class MaxRetriesExceededError(Exception):
    """Raised when the maximum number of retries is exceeded."""

    pass


class RetryHandler(
    EventHandler[Union[AttemptStart, ToolExecutionFailed, InferenceFailed]]
):
    """
    Handles the retry logic for an inference process.

    Listens for AttemptStart to count attempts, and for ToolError/InferenceError
    to trigger retries. Raises MaxRetriesExceededError when the limit is reached.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._current_attempt = 0
        self._retries_used = 0

    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        return (AttemptStart, ToolExecutionFailed, InferenceFailed)

    async def handle(
        self, event: Union[AttemptStart, ToolExecutionFailed, InferenceFailed]
    ) -> None:
        if isinstance(event, AttemptStart):
            self._current_attempt += 1
        elif isinstance(event, (ToolExecutionFailed, InferenceFailed)):
            if self._retries_used < self.max_retries:
                self._retries_used += 1
                raise RequestInferenceRetry()
            else:
                raise MaxRetriesExceededError(
                    f"Failed after {self.max_retries} retries."
                ) from event.error
