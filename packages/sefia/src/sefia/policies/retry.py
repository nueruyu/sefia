from dataclasses import dataclass

from sefia.handlers.retry import RetryHandler
from sefia.interfaces import EventHandler, Policy


@dataclass
class MaxRetries(Policy):
    """A policy that specifies the maximum number of retries on failure."""

    count: int

    def create_handlers(self) -> list[EventHandler]:
        return [RetryHandler(max_retries=self.count)]
