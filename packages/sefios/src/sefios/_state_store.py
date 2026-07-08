from __future__ import annotations

from typing import Generic, Type, TypeVar

from .stores import SessionStore

T = TypeVar("T")


class StateStore(Generic[T]):
    """
    Provides a type-safe, caching wrapper around a SessionStore for a specific key and state type.
    """

    def __init__(self, store: SessionStore, key: str, state_type: Type[T]):
        self._store = store
        self._key = key
        self._state_type = state_type
        self._cache: T | None = None
        self._is_loaded = False

    @property
    def state_type(self) -> Type[T]:
        """The state type this store was created for."""
        return self._state_type

    async def ensure(self) -> T:
        """
        Ensures the state is loaded. If it doesn't exist, returns a
        default-initialized instance. Result is cached for subsequent calls.
        """
        if self._is_loaded:
            if self._cache is None:
                self._cache = self._state_type()
            return self._cache

        state = await self._store.get(self._key, self._state_type)
        if state is None:
            state = self._state_type()
        self._cache = state
        self._is_loaded = True
        return state

    async def get(self, default: T | None = None) -> T | None:
        """
        Returns the state if it exists, otherwise default (None if not specified).
        Result is cached for subsequent calls.
        """
        if self._is_loaded:
            return self._cache if self._cache is not None else default

        self._cache = await self._store.get(self._key, self._state_type)
        self._is_loaded = True
        return self._cache if self._cache is not None else default

    async def save(self, state: T) -> None:
        """Saves the new state and updates the cache."""
        if not isinstance(state, self._state_type):
            raise TypeError(f"State must be an instance of {self._state_type.__name__}")
        await self._store.set(self._key, state, self._state_type)
        self._cache = state
        self._is_loaded = True

    async def delete(self) -> None:
        """Deletes the state and clears the cache."""
        await self._store.delete(self._key)
        self._cache = None
        self._is_loaded = True
