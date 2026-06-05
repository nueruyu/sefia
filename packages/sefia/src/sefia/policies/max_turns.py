from dataclasses import dataclass

from sefia.handlers.max_turns import MaxTurnsHandler
from sefia.interfaces import EventHandler, Policy


@dataclass
class MaxTurns(Policy):
    """A policy that specifies the maximum number of inference loop turns."""

    count: int = 25

    def create_handlers(self) -> list[EventHandler]:
        return [MaxTurnsHandler(max_turns=self.count)]
