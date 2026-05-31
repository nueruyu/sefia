from sefia.handlers.stagnation import StagnationDetector
from sefia.interfaces import EventHandler, Policy


class StagnationPolicy(Policy):
    """
    A policy that adds a handler to detect and prevent infinite loops
    of the same tool call.
    """

    def create_handlers(self) -> list[EventHandler]:
        """Creates a new StagnationDetector instance."""
        return [StagnationDetector()]
