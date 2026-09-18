from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from .exceptions import (
    ExecutionAlreadyTerminalError,
    ExecutionConflictError,
    UnknownExecutionError,
)
from .persistence import PersistenceProvider
from .storage import SessionStorage

InputT = TypeVar("InputT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ExecutionRef:
    """Opaque stable reference to one durable Sefia/Glyff execution session."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("Execution reference must not be empty.")

    @classmethod
    def new(cls) -> ExecutionRef:
        return cls(str(uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class _Cancellation:
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionSnapshot(Generic[InputT, ResultT]):
    ref: ExecutionRef
    input: InputT
    status: Literal["active", "completed", "cancelled"]
    result: ResultT | None = None
    cancellation_reason: str | None = None


class DurableExecutionStore(Generic[InputT, ResultT]):
    """Owns immutable invocation input and terminal state for one execution.

    The execution reference is also the Sefia/Glyff session identifier. An
    application may reopen the same reference after a process restart without
    reconstructing invocation arguments itself.
    """

    def __init__(
        self,
        persistence: PersistenceProvider,
        *,
        input_type: type[InputT],
        result_type: type[ResultT],
        namespace: str = "execution",
    ) -> None:
        namespace = namespace.strip("/")
        if not namespace:
            raise ValueError("Execution namespace must not be empty.")
        self._persistence = persistence
        self._input_type = input_type
        self._result_type = result_type
        self._namespace = namespace

    def new_ref(self) -> ExecutionRef:
        return ExecutionRef.new()

    async def create(self, input_value: InputT) -> ExecutionRef:
        ref = self.new_ref()
        await self.initialize(ref, input_value)
        return ref

    async def initialize(self, ref: ExecutionRef, input_value: InputT) -> None:
        registry = self._persistence.create_session_registry()
        registry.register_session(str(ref))
        storage = self._persistence.create_session_storage(str(ref))
        inserted = await storage.set_if_absent(
            self._input_key,
            input_value,
            self._input_type,
        )
        if inserted:
            return
        existing = await storage.get(self._input_key, self._input_type)
        if existing != input_value:
            raise ExecutionConflictError(str(ref))

    async def input(self, ref: ExecutionRef) -> InputT:
        storage = self._storage(ref)
        value = await storage.get(self._input_key, self._input_type)
        if value is None:
            raise UnknownExecutionError(str(ref))
        return value

    async def complete(self, ref: ExecutionRef, result: ResultT) -> ResultT:
        storage = self._storage(ref)
        await self.input(ref)
        if await storage.get(self._cancel_key, _Cancellation) is not None:
            raise ExecutionAlreadyTerminalError(str(ref))

        inserted = await storage.set_if_absent(
            self._result_key,
            result,
            self._result_type,
        )
        if inserted:
            return result

        existing = await storage.get(self._result_key, self._result_type)
        if existing == result:
            return result
        raise ExecutionConflictError(str(ref))

    async def cancel(self, ref: ExecutionRef, reason: str | None = None) -> None:
        storage = self._storage(ref)
        await self.input(ref)
        if await storage.get(self._result_key, self._result_type) is not None:
            raise ExecutionAlreadyTerminalError(str(ref))

        cancellation = _Cancellation(reason)
        inserted = await storage.set_if_absent(
            self._cancel_key,
            cancellation,
            _Cancellation,
        )
        if inserted:
            return

        existing = await storage.get(self._cancel_key, _Cancellation)
        if existing != cancellation:
            raise ExecutionAlreadyTerminalError(str(ref))

    async def snapshot(self, ref: ExecutionRef) -> ExecutionSnapshot[InputT, ResultT]:
        storage = self._storage(ref)
        input_value = await self.input(ref)
        result = await storage.get(self._result_key, self._result_type)
        if result is not None:
            return ExecutionSnapshot(
                ref=ref,
                input=input_value,
                status="completed",
                result=result,
            )

        cancellation = await storage.get(self._cancel_key, _Cancellation)
        if cancellation is not None:
            return ExecutionSnapshot(
                ref=ref,
                input=input_value,
                status="cancelled",
                cancellation_reason=cancellation.reason,
            )

        return ExecutionSnapshot(ref=ref, input=input_value, status="active")

    def storage(self, ref: ExecutionRef) -> SessionStorage:
        """Return execution-scoped storage for generic execution facilities.

        Lifecycle ownership remains with this execution reference; applications
        should prefer typed facilities such as ``ExternalActionChannel`` over
        inventing their own routing registries.
        """
        return self._storage(ref)

    def _storage(self, ref: ExecutionRef) -> SessionStorage:
        registry = self._persistence.create_session_registry()
        if not registry.session_exists(str(ref)):
            raise UnknownExecutionError(str(ref))
        return self._persistence.create_session_storage(str(ref))

    @property
    def _input_key(self) -> str:
        return f"{self._namespace}/input"

    @property
    def _result_key(self) -> str:
        return f"{self._namespace}/result"

    @property
    def _cancel_key(self) -> str:
        return f"{self._namespace}/cancel"
