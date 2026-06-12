from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._interfaces.event_handler import EventHandler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """Base class for all events."""


class EventPublisher:
    """Manages event handlers and dispatches events."""

    def __init__(self, handlers: list[EventHandler]):
        self._handler_map = self._resolve_handler_map(handlers)

    def _resolve_handler_map(
        self, handlers: list[EventHandler]
    ) -> dict[type[Event], list[EventHandler]]:
        handler_map: dict[type[Event], list[EventHandler]] = defaultdict(list)
        for handler in handlers:
            for event_type in handler.event_types:
                handler_map[event_type].append(handler)
        return handler_map

    async def publish(self, event: Event) -> None:
        event_type = type(event)
        handlers_to_run = self._handler_map.get(event_type, []) + self._handler_map.get(
            Event, []
        )
        for handler in handlers_to_run:
            try:
                await handler.handle(event)
            except Exception:
                logger.exception(
                    "Event handler %s raised while handling %s; ignoring.",
                    type(handler).__name__,
                    event_type.__name__,
                )


__all__ = ["Event", "EventPublisher"]
