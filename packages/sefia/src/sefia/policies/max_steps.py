from dataclasses import dataclass

from sefia.handlers.max_steps import MaxStepsHandler
from sefia.interfaces import EventHandler, Policy


@dataclass
class MaxSteps(Policy):
    """A policy that caps the number of steps in a single inference loop."""

    count: int = 25

    def create_handlers(self) -> list[EventHandler]:
        return [MaxStepsHandler(max_steps=self.count)]
