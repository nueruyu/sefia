import pytest

from sefios import (
    ExecutionManager,
    ExternalActionConflictError,
    ExternalActionDelivery,
    MemoryPersistence,
    UnknownExternalActionError,
)


async def _channel():
    manager = ExecutionManager(MemoryPersistence())
    ref = await manager.create({"message": "hello"}, dict)
    return manager.actions(ref)


async def test_pending_and_result_are_execution_owned():
    channel = await _channel()
    await channel.request("a1", "lookup", {"query": "necra"})

    pending = await channel.pending()
    assert [(item.action_id, item.name) for item in pending] == [("a1", "lookup")]

    assert (
        await channel.accept_result("a1", {"ok": True})
        == ExternalActionDelivery.ACCEPTED
    )
    assert await channel.pending() == []
    assert await channel.result("a1") == {"ok": True}


async def test_same_result_is_retry_safe_but_conflicting_result_is_rejected():
    channel = await _channel()
    await channel.request("a1", "lookup", {})

    assert (
        await channel.accept_result("a1", {"value": 1})
        == ExternalActionDelivery.ACCEPTED
    )
    assert (
        await channel.accept_result("a1", {"value": 1})
        == ExternalActionDelivery.ALREADY_ACCEPTED
    )

    with pytest.raises(ExternalActionConflictError):
        await channel.accept_result("a1", {"value": 2})


async def test_unknown_action_result_is_rejected():
    channel = await _channel()

    with pytest.raises(UnknownExternalActionError):
        await channel.accept_result("missing", None)
