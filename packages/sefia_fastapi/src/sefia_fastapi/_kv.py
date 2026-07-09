from typing import Any, Protocol


class KeyValueStore(Protocol):
    """Async key-value persistence required by the HTTP input state.

    Structurally matches ``sefios.SessionStorage``, so a bound session storage
    can be passed in directly; any other implementation with the same shape
    works too.
    """

    async def get(self, key: str, type_hint: type) -> Any | None: ...

    async def set(self, key: str, value: Any, type_hint: type) -> None: ...

    async def delete(self, key: str) -> None: ...
