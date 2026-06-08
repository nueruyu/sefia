import logging
from collections import defaultdict

from glyff.exceptions import YieldException

from . import events
from .interfaces.event_handler import EventHandler

logger = logging.getLogger(__name__)


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
        """
        Dispatches an event to all handlers registered for its type.

        Event handlers are observers: a misbehaving handler must not break the
        core inference loop, so any exception it raises is logged and swallowed.
        The sole exception is ``YieldException`` — glyff's protocol signal for a
        graceful, resumable interrupt — which is allowed to propagate.
        """
        event_type = type(event)
        handlers_to_run = self._handler_map.get(event_type, []) + self._handler_map.get(
            events.Event, []
        )
        for handler in handlers_to_run:
            try:
                await handler.handle(event)
            except YieldException:
                raise
            except Exception:
                logger.exception(
                    "Event handler %s raised while handling %s; ignoring.",
                    type(handler).__name__,
                    event_type.__name__,
                )
