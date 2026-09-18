from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from ._state_store import StateStore
from .persistence import PersistenceProvider

T = TypeVar("T")

_INPUT_KEY = "sefios/execution/input"
_RESULT_KEY = "sefios/execution/result"
_CANCELLED_KEY = "sefios/execution/cancelled"


class UnknownExecutionError(Exception):
    def __init__(self, execution_id: str) -> None:
        super().__init__(f"Unknown execution: {execution_id}")
        self.execution_id = execution_id


class ExecutionConflictError(Exception):
    """A terminal execution value conflicts with an already accepted value."""


@dataclass(frozen=True)
class ExecutionRef:
    """Opaque reference to one durable Sefia/Glyff inference execution."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("Execution reference must not be empty.")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ExecutionCompleted(Generic[T]):
    result: T


@dataclass(frozen=True)
class ExecutionCancelled:
    pass


ExecutionTerminal = ExecutionCompleted[T] | ExecutionCancelled


class ExecutionManager:
    """Owns durable execution identity, immutable input, and terminal state.

    One execution maps to one Sefia/Glyff session. Applications may attach
    execution-scoped state through :meth:`state`, but they do not need to
    maintain a parallel registry merely to resume an invocation.
    """

    def __init__(self, persistence: PersistenceProvider) -> None:
        self._persistence = persistence
        self._registry = persistence.create_session_registry()

    async def create(self, value: T, value_type: type[T]) -> ExecutionRef:
        execution_id = self._registry.create_session()
        ref = ExecutionRef(execution_id)
        await self._storage(ref).set(_INPUT_KEY, value, value_type)
        return ref

    def exists(self, ref: ExecutionRef) -> bool:
        return self._registry.session_exists(str(ref))

    async def input(self, ref: ExecutionRef, value_type: type[T]) -> T:
        storage = self._require_storage(ref)
        value = await storage.get(_INPUT_KEY, value_type)
        if value is None:
            raise UnknownExecutionError(str(ref))
        return value

    def state(self, ref: ExecutionRef, key: str, state_type: type[T]) -> StateStore[T]:
        key = key.strip("/")
        if not key:
            raise ValueError("Execution state key must not be empty.")
        return StateStore(
            storage=self._require_storage(ref),
            key=f"sefios/execution/state/{key}",
            state_type=state_type,
        )

    async def complete(
        self,
        ref: ExecutionRef,
        result: T,
        result_type: type[T],
    ) -> ExecutionCompleted[T]:
        storage = self._require_storage(ref)
        if await storage.get(_CANCELLED_KEY, bool):
            raise ExecutionConflictError("Execution is already cancelled.")
        existing = await storage.get(_RESULT_KEY, result_type)
        if existing is not None:
            if existing != result:
                raise ExecutionConflictError(
                    "Execution already completed with a different result."
                )
            return ExecutionCompleted(existing)
        await storage.set(_RESULT_KEY, result, result_type)
        return ExecutionCompleted(result)

    async def cancel(self, ref: ExecutionRef) -> ExecutionCancelled:
        storage = self._require_storage(ref)
        if await storage.get(_RESULT_KEY, object) is not None:
            raise ExecutionConflictError("Execution is already completed.")
        await storage.set(_CANCELLED_KEY, True, bool)
        return ExecutionCancelled()

    async def terminal(
        self,
        ref: ExecutionRef,
        result_type: type[T],
    ) -> ExecutionTerminal[T] | None:
        storage = self._require_storage(ref)
        result = await storage.get(_RESULT_KEY, result_type)
        if result is not None:
            return ExecutionCompleted(result)
        if await storage.get(_CANCELLED_KEY, bool):
            return ExecutionCancelled()
        return None

    def actions(self, ref: ExecutionRef):
        from ._external_action import ExternalActionChannel

        return ExternalActionChannel(self._require_storage(ref))

    def _require_storage(self, ref: ExecutionRef):
        if not self.exists(ref):
            raise UnknownExecutionError(str(ref))
        return self._storage(ref)

    def _storage(self, ref: ExecutionRef):
        return self._persistence.create_session_storage(str(ref))
