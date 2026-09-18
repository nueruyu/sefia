from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .exceptions import ExternalActionConflictError, UnknownExternalActionError
from .storage import SessionStorage

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class ExternalActionRequest(Generic[RequestT]):
    id: str
    payload: RequestT


class ExternalActionChannel(Generic[RequestT, ResultT]):
    """Durable external-action routing owned by the execution layer.

    Requests and results are immutable. Re-delivering the same result is
    idempotent; delivering a different result for the same action is rejected.
    Pending requests are discovered from persisted request keys, so applications
    do not maintain a parallel action catalog.
    """

    def __init__(
        self,
        storage: SessionStorage,
        *,
        request_type: type[RequestT],
        result_type: type[ResultT],
        namespace: str = "external_actions",
    ) -> None:
        namespace = namespace.strip("/")
        if not namespace:
            raise ValueError("External action namespace must not be empty.")
        self._storage = storage
        self._request_type = request_type
        self._result_type = result_type
        self._namespace = namespace

    async def record_request(self, action_id: str, request: RequestT) -> None:
        self._validate_id(action_id)
        key = self._request_key(action_id)
        inserted = await self._storage.set_if_absent(
            key,
            request,
            self._request_type,
        )
        if inserted:
            return
        existing = await self._storage.get(key, self._request_type)
        if existing != request:
            raise ExternalActionConflictError(action_id)

    async def request(self, action_id: str) -> RequestT | None:
        self._validate_id(action_id)
        return await self._storage.get(
            self._request_key(action_id),
            self._request_type,
        )

    async def pending(self) -> tuple[ExternalActionRequest[RequestT], ...]:
        keys = await self._storage.keys(self._request_prefix)
        pending: list[ExternalActionRequest[RequestT]] = []
        for key in keys:
            action_id = key[len(self._request_prefix) :]
            request = await self.request(action_id)
            if request is None:
                continue
            if await self.result(action_id) is None:
                pending.append(ExternalActionRequest(action_id, request))
        return tuple(pending)

    async def accept_result(self, action_id: str, result: ResultT) -> ResultT:
        self._validate_id(action_id)
        if await self.request(action_id) is None:
            raise UnknownExternalActionError(action_id)

        key = self._result_key(action_id)
        inserted = await self._storage.set_if_absent(
            key,
            result,
            self._result_type,
        )
        if inserted:
            return result

        existing = await self._storage.get(key, self._result_type)
        if existing == result:
            return result
        raise ExternalActionConflictError(action_id)

    async def result(self, action_id: str) -> ResultT | None:
        self._validate_id(action_id)
        return await self._storage.get(
            self._result_key(action_id),
            self._result_type,
        )

    @staticmethod
    def _validate_id(action_id: str) -> None:
        if not action_id or "/" in action_id:
            raise ValueError("External action id must be a non-empty path segment.")

    @property
    def _request_prefix(self) -> str:
        return f"{self._namespace}/request/"

    def _request_key(self, action_id: str) -> str:
        return f"{self._request_prefix}{action_id}"

    def _result_key(self, action_id: str) -> str:
        return f"{self._namespace}/result/{action_id}"
