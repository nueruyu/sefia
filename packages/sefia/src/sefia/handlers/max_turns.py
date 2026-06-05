from typing import Type

from sefia.events import Event, NextTurnRequested
from sefia.interfaces import EventHandler


class RequestNextTurn(Exception):
    """Signal raised by a handler to permit the inference loop to take another turn."""

    pass


class MaxTurnsExceededError(Exception):
    """Raised when another inference turn is needed but no handler permits one."""

    pass


class MaxTurnsHandler(EventHandler[NextTurnRequested]):
    """
    Permits the inference loop to continue up to a maximum number of turns.

    The executor owns the turn counter and does not loop on its own; it fires
    ``NextTurnRequested`` before each turn beyond the first. This handler grants
    the next turn by raising ``RequestNextTurn`` while the limit has not been
    reached, and otherwise stays silent so the executor stops the loop.

    ``max_turns`` is the number of turns allowed within a single attempt
    (default ``1``: a single step, no looping). Pass ``None`` for no limit.
    """

    def __init__(self, max_turns: int | None = 1):
        self.max_turns = max_turns

    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        return (NextTurnRequested,)

    async def handle(self, event: NextTurnRequested) -> None:
        if self.max_turns is None or event.completed_turns < self.max_turns:
            raise RequestNextTurn()
