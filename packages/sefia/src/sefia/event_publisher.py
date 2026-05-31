from collections import defaultdict

from . import events
from .interfaces.event_handler import EventHandler


class EventPublisher:
    """Manages event handlers and dispatches events."""

    def __init__(self, handlers: list[EventHandler]):
        self._handler_map = self._resolve_handler_map(handlers)

    def _resolve_handler_map(
        self, handlers: list[EventHandler]
    ) -> dict[type[events.Event], list[EventHandler]]:
        """Inspects handlers to map event types to the handlers that process them."""
        handler_map: dict[type[events.Event], list[EventHandler]] = defaultdict(list)
        for handler in handlers:
            for event_type in handler.event_types:
                handler_map[event_type].append(handler)
        return handler_map

    async def publish(self, event: events.Event) -> None:
        """Dispatches an event to all handlers registered for its type."""
        event_type = type(event)
        handlers_to_run = self._handler_map.get(event_type, []) + self._handler_map.get(
            events.Event, []
        )
        for handler in handlers_to_run:
            await handler.handle(event)
