"""A type-keyed state container for sefios.

Handlers and applications should not hand-roll string keys when persisting
state. Instead they register a state type once with :func:`state` and then
retrieve a :class:`~sefia._state_store.StateStore` for it via the
:class:`StateContainer`, keyed by the type itself. The container does not know
about any individual state (such as cost); it only resolves the registered
string key and delegates persistence to the underlying sefia ``SessionStore``.
"""

from __future__ import annotations

from typing import Type, TypeVar

from sefia import SessionContext, StateStore, get_context

T = TypeVar("T")


class StateRegistry:
    """Maps state types to the string keys used for their persistence.

    This is the single place where the type -> persistence-key relationship
    lives, keeping string keys out of handlers and application code.
    """

    def __init__(self) -> None:
        self._keys: dict[type, str] = {}

    def register(self, state_type: type, key: str) -> None:
        """Registers ``state_type`` under ``key``.

        A type may only be registered once, and a key may only be claimed by a
        single type; violating either raises ``ValueError`` to surface
        collisions early.
        """
        if state_type in self._keys:
            raise ValueError(
                f"{state_type.__module__}.{state_type.__name__} is already "
                f"registered under key {self._keys[state_type]!r}."
            )
        for other_type, other_key in self._keys.items():
            if other_key == key:
                raise ValueError(
                    f"Key {key!r} is already registered for "
                    f"{other_type.__module__}.{other_type.__name__}."
                )
        self._keys[state_type] = key

    def key_for(self, state_type: type) -> str:
        """Returns the registered key for ``state_type``.

        Raises ``KeyError`` if the type was never registered with ``@state``.
        """
        try:
            return self._keys[state_type]
        except KeyError:
            raise KeyError(
                f"{state_type.__module__}.{state_type.__name__} is not a "
                f"registered state type. Decorate it with @state(key=...) "
                f"before using it."
            ) from None


_default_registry = StateRegistry()


def state(key: str):
    """Class decorator registering a state type under a persistence ``key``.

    Apply it outside ``@dataclass`` so the already-built type is registered::

        @state(key="sefios.cost")
        @dataclass(frozen=True)
        class CostState:
            cost: float = 0.0
    """

    def decorator(cls: type) -> type:
        _default_registry.register(cls, key)
        return cls

    return decorator


class StateContainer:
    """Retrieves per-type state stores, keyed by the state type itself.

    The container is generic: it has no knowledge of individual states. It
    resolves the registered string key for a type and delegates to the
    session's :meth:`SessionContext.get_state_store`, reusing the existing
    sefia persistence machinery.
    """

    def __init__(
        self,
        ctx: SessionContext,
        registry: StateRegistry = _default_registry,
    ) -> None:
        self._ctx = ctx
        self._registry = registry

    def get(self, state_type: Type[T]) -> StateStore[T]:
        """Returns the ``StateStore`` for ``state_type``."""
        key = self._registry.key_for(state_type)
        return self._ctx.get_state_store(key, state_type)


def get_state() -> StateContainer:
    """Returns a :class:`StateContainer` bound to the current session context."""
    return StateContainer(get_context())


__all__ = [
    "StateRegistry",
    "StateContainer",
    "state",
    "get_state",
]
