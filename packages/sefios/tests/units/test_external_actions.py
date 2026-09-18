from dataclasses import dataclass

import pytest
from glyff_pydantic import PydanticSerializer
from sefios import ExternalActionChannel, MemorySessionStorage
from sefios.exceptions import (
    ExternalActionConflictError,
    UnknownExternalActionError,
)


@dataclass(frozen=True)
class Request:
    tool: str


@dataclass(frozen=True)
class Result:
    value: str


@pytest.fixture
def channel() -> ExternalActionChannel[Request, Result]:
    return ExternalActionChannel(
        MemorySessionStorage(PydanticSerializer()),
        request_type=Request,
        result_type=Result,
    )


async def test_pending_is_derived_from_persisted_requests(channel):
    await channel.record_request("b", Request("second"))
    await channel.record_request("a", Request("first"))

    assert [item.id for item in await channel.pending()] == ["a", "b"]

    await channel.accept_result("a", Result("done"))

    assert [item.id for item in await channel.pending()] == ["b"]


async def test_same_result_is_idempotent_and_conflicting_result_is_rejected(channel):
    await channel.record_request("action", Request("tool"))

    first = await channel.accept_result("action", Result("same"))
    second = await channel.accept_result("action", Result("same"))

    assert first == second == Result("same")

    with pytest.raises(ExternalActionConflictError):
        await channel.accept_result("action", Result("different"))


async def test_result_requires_a_recorded_request(channel):
    with pytest.raises(UnknownExternalActionError):
        await channel.accept_result("missing", Result("value"))


async def test_request_identity_is_immutable(channel):
    await channel.record_request("action", Request("first"))
    await channel.record_request("action", Request("first"))

    with pytest.raises(ExternalActionConflictError):
        await channel.record_request("action", Request("different"))
