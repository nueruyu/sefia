from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .storage import SessionStorage

_INDEX_KEY = "sefios/external_actions/index"


class UnknownExternalActionError(Exception):
    def __init__(self, action_id: str) -> None:
        super().__init__(f"Unknown external action: {action_id}")
        self.action_id = action_id


class ExternalActionConflictError(Exception):
    """A repeated action delivery conflicts with the durable accepted value."""


class ExternalActionDelivery(str, Enum):
    ACCEPTED = "accepted"
    ALREADY_ACCEPTED = "already_accepted"


class _StoredRequest(BaseModel):
    action_id: str
    name: str
    arguments: dict[str, Any]


class _StoredResult(BaseModel):
    value: Any


class _ActionIndex(BaseModel):
    action_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ExternalActionRequest:
    action_id: str
    name: str
    arguments: dict[str, Any]


class ExternalActionChannel:
    """Durable external-action requests and idempotent result delivery.

    This is execution infrastructure: request persistence, pending discovery,
    duplicate delivery, and replay-visible results are owned here rather than
    by host applications.
    """

    def __init__(self, storage: SessionStorage) -> None:
        self._storage = storage

    async def request(
        self,
        action_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ExternalActionRequest:
        candidate = _StoredRequest(
            action_id=action_id,
            name=name,
            arguments=arguments,
        )
        key = self._request_key(action_id)
        existing = await self._storage.get(key, _StoredRequest)
        if existing is not None and existing != candidate:
            raise ExternalActionConflictError(
                f"External action {action_id} was already recorded differently."
            )
        if existing is None:
            await self._storage.set(key, candidate, _StoredRequest)
            index = await self._index()
            if action_id not in index.action_ids:
                index.action_ids.append(action_id)
                await self._storage.set(_INDEX_KEY, index, _ActionIndex)
        return self._to_request(candidate)

    async def pending(self) -> list[ExternalActionRequest]:
        index = await self._index()
        pending: list[ExternalActionRequest] = []
        for action_id in index.action_ids:
            request = await self._storage.get(
                self._request_key(action_id),
                _StoredRequest,
            )
            if request is None:
                continue
            if await self._storage.get(self._result_key(action_id), _StoredResult) is None:
                pending.append(self._to_request(request))
        return pending

    async def accept_result(
        self,
        action_id: str,
        value: Any,
    ) -> ExternalActionDelivery:
        request = await self._storage.get(
            self._request_key(action_id),
            _StoredRequest,
        )
        if request is None:
            raise UnknownExternalActionError(action_id)

        key = self._result_key(action_id)
        existing = await self._storage.get(key, _StoredResult)
        candidate = _StoredResult(value=value)
        if existing is not None:
            if existing != candidate:
                raise ExternalActionConflictError(
                    f"External action {action_id} already has a different result."
                )
            return ExternalActionDelivery.ALREADY_ACCEPTED

        await self._storage.set(key, candidate, _StoredResult)
        return ExternalActionDelivery.ACCEPTED

    async def result(self, action_id: str) -> Any | None:
        stored = await self._storage.get(
            self._result_key(action_id),
            _StoredResult,
        )
        return None if stored is None else stored.value

    async def get(self, action_id: str) -> ExternalActionRequest:
        stored = await self._storage.get(
            self._request_key(action_id),
            _StoredRequest,
        )
        if stored is None:
            raise UnknownExternalActionError(action_id)
        return self._to_request(stored)

    async def _index(self) -> _ActionIndex:
        stored = await self._storage.get(_INDEX_KEY, _ActionIndex)
        return _ActionIndex() if stored is None else stored

    @staticmethod
    def _request_key(action_id: str) -> str:
        return f"sefios/external_actions/request/{action_id}"

    @staticmethod
    def _result_key(action_id: str) -> str:
        return f"sefios/external_actions/result/{action_id}"

    @staticmethod
    def _to_request(stored: _StoredRequest) -> ExternalActionRequest:
        return ExternalActionRequest(
            action_id=stored.action_id,
            name=stored.name,
            arguments=dict(stored.arguments),
        )
