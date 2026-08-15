from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from types import UnionType
from typing import Any, Generic, Type, TypeVar, Union, get_args, get_origin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """Base class for all events."""


E = TypeVar("E", bound=Event)


def _substitute_typevars(annotation: object, typevars: dict[TypeVar, object]) -> object:
    if isinstance(annotation, TypeVar):
        return typevars.get(annotation, annotation)

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return tuple(
            _substitute_typevars(arg, typevars) for arg in get_args(annotation)
        )

    return annotation


def _event_types_from_annotation(annotation: object) -> tuple[Type[Event], ...] | None:
    if isinstance(annotation, tuple):
        args = annotation
    else:
        origin = get_origin(annotation)
        args = get_args(annotation) if origin in (Union, UnionType) else (annotation,)

    event_types: list[Type[Event]] = []
    for arg in args:
        event_type = get_origin(arg) or arg
        if not isinstance(event_type, type) or not issubclass(event_type, Event):
            return None
        event_types.append(event_type)
    return tuple(event_types)


def _infer_event_types_from_bases(
    handler_cls: type[EventHandler[Any]], typevars: dict[TypeVar, object]
) -> tuple[Type[Event], ...] | None:
    for base in getattr(handler_cls, "__orig_bases__", ()):
        origin = get_origin(base)
        if origin is None:
            continue

        args = tuple(_substitute_typevars(arg, typevars) for arg in get_args(base))
        if origin is EventHandler:
            if not args:
                return None
            return _event_types_from_annotation(args[0])

        if isinstance(origin, type) and issubclass(origin, EventHandler):
            parameters = getattr(origin, "__parameters__", ())
            next_typevars = dict(typevars)
            next_typevars.update(dict(zip(parameters, args)))
            event_types = _infer_event_types_from_bases(origin, next_typevars)
            if event_types is not None:
                return event_types

    return None


@lru_cache(maxsize=None)
def _infer_event_types(handler_cls: type[EventHandler[Any]]) -> tuple[Type[Event], ...]:
    all_event_types: list[Type[Event]] = []
    for cls in handler_cls.__mro__:
        event_types = _infer_event_types_from_bases(cls, {})
        if event_types is not None:
            all_event_types.extend(event_types)

    if all_event_types:
        seen: set[Type[Event]] = set()
        return tuple(
            event_type
            for event_type in all_event_types
            if not (event_type in seen or seen.add(event_type))
        )

    raise TypeError(
        f"{handler_cls.__name__} must specify concrete EventHandler[...] event "
        "type arguments or override event_types."
    )


class EventHandler(ABC, Generic[E]):
    """
    Abstract base class for a handler that processes a specific type of event.
    """

    @property
    def event_types(self) -> tuple[Type[Event], ...]:
        """Returns a tuple of event types that this handler can process."""
        return _infer_event_types(type(self))

    @abstractmethod
    async def handle(self, event: E) -> None:
        """Process a specific inference event."""
        ...


class EventPublisher:
    """Manages event handlers and dispatches events."""

    def __init__(self, handlers: list[EventHandler[Any]]):
        self._handler_map = self._resolve_handler_map(handlers)

    def _resolve_handler_map(
        self, handlers: list[EventHandler[Any]]
    ) -> dict[type[Event], list[EventHandler[Any]]]:
        """Inspects handlers to map event types to the handlers that process them."""
        handler_map: dict[type[Event], list[EventHandler[Any]]] = defaultdict(list)
        for handler in handlers:
            for event_type in handler.event_types:
                handler_map[event_type].append(handler)
        return handler_map

    async def publish(self, event: Event) -> None:
        """
        Dispatches an event to all handlers registered for its type.

        Event handlers are pure observers: they cannot steer the inference loop.
        Any exception a handler raises — including ``PauseException`` — is logged
        and swallowed here, so a misbehaving observer can never affect control
        flow. Genuine resumable interrupts are driven by the control/execution
        layer (for example, a tool raising ``PauseException``), never by an
        observer.
        """
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


__all__ = ["Event", "EventPublisher", "EventHandler"]
